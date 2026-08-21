"""Model to infer mass spectra from SMILES.

This module contains the EIMSPredictor class, which is a joint model that
combines a FragmentGenerator (or full enumeration) and an IntensityPredictor to
infer mass spectra from SMILES.
"""

import csv
import gc
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

import dgl
import numpy as np
import torch
import torch.utils.data
from rdkit.Chem import Descriptors

from icicle.data.datasets.smiles_inference import (
    RootGraphInferenceDataset,
    root_only_collate_fn,
)
from icicle.data.fragmentation_engine import (
    FragmentationParams,
    FragmentEngine,
)
from icicle.data.gpu_fragment_enumeration import (
    enumerate_fragments_gpu,
    enumerate_fragments_gpu_batched,
    populate_frag_to_entry_from_gpu,
)
from icicle.data.tree_processing import TreeProcessingConfig, TreeProcessor
from icicle.models.base_model import BaseSpectrumPredictor
from icicle.models.fragmentation_model import FragmentGenerator
from icicle.models.intensity_model import IntensityPredictor


def _dataloader_worker_init(worker_id):
    """Suppress RDKit and C++ warnings in each DataLoader worker process."""
    import os as _os
    import warnings

    import torch.multiprocessing as _mp

    _mp.set_sharing_strategy("file_system")
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass
    _os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    warnings.filterwarnings("ignore", message=".*lazyInitCUDA.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class _BaseEIMSPredictor(BaseSpectrumPredictor, ABC):
    """Abstract base class for EIMS prediction models."""

    def __init__(
        self,
        min_mz: Optional[float] = None,
        max_mz: Optional[float] = None,
        bin_width: Optional[float] = None,
        intensity_predictor: Optional[IntensityPredictor] = None,
        intensity_predictor_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        # Load checkpoint before super().__init__ so we can infer mz range from it.
        _ip = intensity_predictor
        if _ip is None and intensity_predictor_checkpoint is not None:
            _ip = IntensityPredictor.load_from_checkpoint(
                intensity_predictor_checkpoint, map_location="cpu"
            )

        # Infer mz range from checkpoint when not explicitly provided.
        if _ip is not None:
            min_mz = min_mz if min_mz is not None else _ip.min_mz
            max_mz = max_mz if max_mz is not None else _ip.max_mz
            bin_width = bin_width if bin_width is not None else _ip.bin_width

        if min_mz is None or max_mz is None or bin_width is None:
            raise ValueError(
                "min_mz, max_mz, bin_width must be provided if no checkpoint is given."
            )

        super().__init__(
            min_mz=min_mz, max_mz=max_mz, bin_width=bin_width, **kwargs
        )

        self.intensity_predictor: Optional[IntensityPredictor] = _ip
        self.tree_processor: Optional[TreeProcessor] = None

        if self.intensity_predictor is not None:
            self._init_tree_processor_from_intensity_predictor()

    def _init_tree_processor_from_intensity_predictor(self):
        """Initialize TreeProcessor from intensity predictor config."""
        ip = self.intensity_predictor
        config = TreeProcessingConfig(
            pe_embed_k=ip.pe_embed_k,
            add_hs=ip.add_hs,
            embed_elem_group=ip.embed_elem_group,
            min_mz=ip.min_mz,
            max_mz=ip.max_mz,
            bin_width=ip.bin_width,
            max_broken_bonds=6,  # FIXME: infer from dataset config
            max_tree_depth=3,
            num_h_shifts=ip.h_shift_range,
            hetero_weights_cc=2,
            hetero_weights_other=1,
        )
        self.tree_processor = TreeProcessor(config)

    @abstractmethod
    def _generate_fragments(self, smiles: str, device: str, **kwargs) -> Any:
        pass

    @abstractmethod
    def _fragments_to_intensity_inputs(
        self, root_smiles: str, fragments_raw_output: Any, device: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def _format_spectrum_result(
        self,
        intensity_result: Dict[str, List[torch.Tensor]],
        fragments_raw_output: Any,
        smiles: str,
    ) -> Dict[str, Any]:
        pass

    def load_from_checkpoint(
        self, intensity_predictor_checkpoint: str, **kwargs
    ) -> None:
        """Load intensity predictor from checkpoint."""
        self.intensity_predictor = IntensityPredictor.load_from_checkpoint(
            intensity_predictor_checkpoint, map_location="cpu"
        )
        self._init_tree_processor_from_intensity_predictor()

    def predict_from_smiles(
        self, smiles: str, device: str = "cpu", **kwargs
    ) -> Dict[str, Any]:
        """Predict a spectrum from a SMILES string."""
        if self.intensity_predictor is None:
            raise ValueError("Intensity predictor not loaded.")
        if self.tree_processor is None:
            self._init_tree_processor_from_intensity_predictor()

        self.intensity_predictor.to(device)
        self.intensity_predictor.eval()
        self._check_and_fix_dimension_mismatch(device)

        fragments_raw_output = self._generate_fragments(
            smiles, device, **kwargs
        )
        if fragments_raw_output is None:
            return self._empty_spectrum_result()

        intensity_inputs = self._fragments_to_intensity_inputs(
            smiles, fragments_raw_output, device
        )
        if intensity_inputs is None:
            return self._empty_spectrum_result()

        try:
            with torch.no_grad():
                intensity_result = (
                    self.intensity_predictor.predict_intensities(
                        graphs=intensity_inputs["graphs"],
                        root_reprs=intensity_inputs["root_reprs"],
                        ind_maps=intensity_inputs["ind_maps"],
                        num_frags=intensity_inputs["num_frags"],
                        broken_bonds=intensity_inputs["broken_bonds"],
                        masses=intensity_inputs["masses"],
                        max_add_hs=intensity_inputs.get("max_add_hs"),
                        max_remove_hs=intensity_inputs.get("max_remove_hs"),
                        root_form_vecs=intensity_inputs.get("root_form_vecs"),
                        frag_form_vecs=intensity_inputs.get("frag_form_vecs"),
                        formulae=intensity_inputs.get("formulae"),
                    )
                )
        except Exception as e:
            # E.g. a disconnected-component SMILES (salt/hydrate, containing
            # ".") can produce a zero-edge root graph, which crashes deep in
            # the GNN encoder with a torch RuntimeError unrelated to memory
            # ("min(): Expected reduction dim to be specified for
            # input.numel() == 0") rather than raising a clean, catchable
            # error earlier. Degrade to the empty-spectrum result, matching
            # how this function already handles other "can't process this
            # molecule" cases above, instead of propagating a crash.
            logging.warning(
                f"Intensity prediction failed for {smiles!r}: {e} — "
                "returning empty spectrum."
            )
            return self._empty_spectrum_result()

        return self._format_spectrum_result(
            intensity_result, fragments_raw_output, smiles
        )

    def _create_root_representation(
        self, root_smiles: str, engine: FragmentEngine, device
    ):
        """Create root molecule DGL graph for the intensity predictor."""
        if isinstance(device, str):
            device = torch.device(device)
        root_frag = engine.get_root_frag()
        root_graph_dict = self.tree_processor.featurize_frag(
            frag=root_frag, engine=engine, add_random_walk=False
        )
        root_graph = root_graph_dict["graph"].to(device)
        self.tree_processor.add_pe_embed(root_graph)
        return root_graph

    def _check_and_fix_dimension_mismatch(self, device: str) -> bool:
        """Check for node feature dimension mismatches between tree processor
        and intensity predictor."""
        if self.tree_processor is None or self.intensity_predictor is None:
            return False

        frag_node_feats = self.tree_processor.get_node_feats()
        inten_node_feats = self.intensity_predictor.node_feats

        if frag_node_feats == inten_node_feats:
            return True
        if self.intensity_predictor.inject_early:
            expected = inten_node_feats - self.intensity_predictor.hidden_size
            if frag_node_feats == expected:
                return True
            logging.warning(
                "Dimension mismatch not explained by early injection"
            )
        else:
            logging.warning(
                f"Node feature mismatch: frag={frag_node_feats}, inten={inten_node_feats}"
            )
        return False

    def _empty_spectrum_result(self) -> Dict[str, Any]:
        """Return a zero-intensity spectrum result."""
        mz_bins = self.intensity_predictor.inten_buckets.cpu().numpy()
        return {
            "smiles": "",
            "mz_bins": mz_bins,
            "intensities": np.zeros_like(mz_bins),
            "num_fragments": 0,
            "fragments": {},
        }

    def _build_intensity_inputs(
        self,
        root_smiles: str,
        engine: FragmentEngine,
        device: str,
        frag_entries_iter,
        get_frag_id,
        get_max_broken,
        get_base_mass,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build the intensity predictor input dict from an iterable of
        fragment entries.

        Shared by both CPU-path subclasses. Callers pass accessors for the
        different entry formats (FragEntry attrs vs dict keys).
        """
        from icicle.utils import formula_from_smi, formula_to_dense

        frag_entries = list(frag_entries_iter)

        h_shifts = torch.arange(
            -self.intensity_predictor.h_shift_range,
            self.intensity_predictor.h_shift_range + 1,
            device=device,
            dtype=torch.float,
        )

        if len(frag_entries) == 0:
            root_form_vec = None
            frag_form_vecs = None
            if self.intensity_predictor.encode_formulae:
                root_form = formula_from_smi(root_smiles)
                root_form_vec = (
                    torch.from_numpy(formula_to_dense(root_form))
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                frag_form_vecs = torch.zeros(
                    1,
                    0,
                    root_form_vec.shape[-1],
                    device=device,
                    dtype=torch.float,
                )
            return {
                "graphs": dgl.graph(([], []), num_nodes=0).to(device),
                "root_reprs": self._create_root_representation(
                    root_smiles, engine, device
                ),
                "ind_maps": torch.zeros(0, dtype=torch.long, device=device),
                "num_frags": torch.tensor(
                    [0], device=device, dtype=torch.long
                ),
                "broken_bonds": torch.zeros(
                    1, 0, device=device, dtype=torch.float
                ),
                "masses": torch.zeros(
                    1,
                    0,
                    1,
                    self.intensity_predictor.mass_bins,
                    device=device,
                    dtype=torch.float,
                ),
                "max_add_hs": torch.zeros(
                    1, 0, device=device, dtype=torch.float
                ),
                "max_remove_hs": torch.zeros(
                    1, 0, device=device, dtype=torch.float
                ),
                "root_form_vecs": root_form_vec,
                "frag_form_vecs": frag_form_vecs,
            }

        frag_graphs, broken_bonds, fragment_masses = [], [], []
        max_add_hs_list, max_remove_hs_list, frag_forms = [], [], []

        # One featurize_frag call per fragment — sequential CPU loop.
        # GPU path replaces this with batch_remove_single_atoms across all fragments at once.
        for entry in frag_entries:
            frag_id = get_frag_id(entry)
            frag_info = self.tree_processor.featurize_frag(
                frag=frag_id, engine=engine, add_random_walk=False
            )
            frag_graphs.append(frag_info["graph"])
            broken_bonds.append(get_max_broken(entry))

            if self.intensity_predictor.encode_formulae:
                from icicle.utils import formula_to_dense

                frag_forms.append(formula_to_dense(frag_info.get("form", "")))

            fragment_masses.append(get_base_mass(entry) + h_shifts)
            max_add_hs_list.append(self.intensity_predictor.h_shift_range)
            max_remove_hs_list.append(self.intensity_predictor.h_shift_range)

        if not frag_graphs:
            return None

        batched_graph = dgl.batch(frag_graphs).to(device)
        self.tree_processor.add_pe_embed(batched_graph)
        root_repr = self._create_root_representation(
            root_smiles, engine, device
        )

        result = {
            "graphs": batched_graph,
            "root_reprs": root_repr,
            "ind_maps": torch.zeros(
                len(frag_graphs), dtype=torch.long, device=device
            ),
            "num_frags": torch.tensor(
                [len(frag_graphs)], device=device, dtype=torch.long
            ),
            "broken_bonds": torch.tensor(
                broken_bonds, dtype=torch.float, device=device
            ).unsqueeze(0),
            "masses": torch.stack(fragment_masses).unsqueeze(0).unsqueeze(2),
            "max_add_hs": torch.tensor(
                max_add_hs_list, dtype=torch.float, device=device
            ).unsqueeze(0),
            "max_remove_hs": torch.tensor(
                max_remove_hs_list, dtype=torch.float, device=device
            ).unsqueeze(0),
        }

        if self.intensity_predictor.encode_formulae and frag_forms:
            root_form = formula_from_smi(root_smiles)
            result["root_form_vecs"] = (
                torch.from_numpy(formula_to_dense(root_form))
                .float()
                .unsqueeze(0)
                .to(device)
            )
            result["frag_form_vecs"] = (
                torch.from_numpy(np.array(frag_forms))
                .float()
                .unsqueeze(0)
                .to(device)
            )

        return result

    def batch_predict_from_smiles(
        self,
        smiles_list: List[str],
        device: str = "cpu",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Predict spectra for a list of SMILES strings sequentially."""
        return [
            self.predict_from_smiles(s, device=device, **kwargs)
            for s in smiles_list
        ]


class EIMSPredictorWithFragmentGenerator(_BaseEIMSPredictor):
    """EIMS predictor using a learned FragmentGenerator model."""

    def __init__(
        self,
        fragment_generator: Optional[FragmentGenerator] = None,
        intensity_predictor: Optional[IntensityPredictor] = None,
        fragment_generator_checkpoint: Optional[str] = None,
        intensity_predictor_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            intensity_predictor=intensity_predictor,
            intensity_predictor_checkpoint=intensity_predictor_checkpoint,
            **kwargs,
        )
        self.fragment_generator: Optional[FragmentGenerator] = None

        if fragment_generator is not None:
            self.fragment_generator = fragment_generator
        elif fragment_generator_checkpoint is not None:
            self.fragment_generator = FragmentGenerator.load_from_checkpoint(
                fragment_generator_checkpoint, map_location="cpu"
            )

        if self.fragment_generator is not None and hasattr(
            self.fragment_generator, "tree_processor"
        ):
            self.tree_processor = self.fragment_generator.tree_processor
            if (
                self.intensity_predictor
                and hasattr(self.intensity_predictor, "tree_processor")
                and self.tree_processor
                != self.intensity_predictor.tree_processor
            ):
                logging.warning(
                    "FragmentGenerator and IntensityPredictor have different TreeProcessors. "
                    "Using FragmentGenerator's tree_processor for consistency."
                )

    def load_from_checkpoint(
        self,
        fragment_generator_checkpoint: str,
        intensity_predictor_checkpoint: str,
    ) -> None:
        """Load both model checkpoints."""
        super().load_from_checkpoint(intensity_predictor_checkpoint)
        self.fragment_generator = FragmentGenerator.load_from_checkpoint(
            fragment_generator_checkpoint, map_location="cpu"
        )
        if not hasattr(self.fragment_generator, "tree_processor"):
            raise ValueError(
                "Loaded FragmentGenerator must have a tree_processor"
            )
        self.tree_processor = self.fragment_generator.tree_processor

    def _generate_fragments(
        self,
        smiles: str,
        device: str,
        max_nodes: int = 100,
        threshold: float = 0.01,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generate fragments using the FragmentGenerator model."""
        if self.fragment_generator is None:
            raise RuntimeError("FragmentGenerator is not loaded.")
        self.fragment_generator.to(device)
        self.fragment_generator.eval()
        return self.fragment_generator.predict_mol(
            smi=smiles,
            device=device,
            max_nodes=max_nodes,
            threshold=threshold,
            **kwargs,
        )

    def _fragments_to_intensity_inputs(
        self, root_smiles: str, fragments_result: Dict[str, Any], device: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Convert FragmentGenerator output to intensity predictor input
        format."""
        engine = FragmentEngine(root_smiles)

        def _get_base_mass(entry):
            mass = entry.get("base_mass", 0.0)
            if mass == 0.0:
                frag_mol = engine.get_frag_mol(entry["frag"])
                mass = Descriptors.ExactMolWt(frag_mol)
            return mass

        return self._build_intensity_inputs(
            root_smiles=root_smiles,
            engine=engine,
            device=device,
            frag_entries_iter=fragments_result.values(),
            get_frag_id=lambda e: e["frag"],
            get_max_broken=lambda e: e.get("max_broken", 0),
            get_base_mass=_get_base_mass,
        )

    def _format_spectrum_result(
        self,
        intensity_result: Dict[str, List[torch.Tensor]],
        fragments_result: Dict[str, Any],
        smiles: str,
    ) -> Dict[str, Any]:
        """Format spectrum result for FragmentGenerator output."""
        intensities = intensity_result["spec"][0].detach().cpu().numpy()
        mz_bins = self.intensity_predictor.inten_buckets.cpu().numpy()

        fragments_for_plotting = {}
        if fragments_result:
            engine = FragmentEngine(smiles)
            for frag_hash, frag_data in fragments_result.items():
                frag_bitmask = frag_data.get("frag")
                base_mass = frag_data.get("base_mass")
                form = frag_data.get("form")
                if frag_bitmask is not None and base_mass is not None and form:
                    draw_info = engine.get_draw_dict(frag_bitmask)
                    fragments_for_plotting[float(base_mass)] = {
                        "structure": draw_info.mol,
                        "form": form,
                        "highlights": {
                            "atoms": draw_info.hatoms,
                            "bonds": draw_info.hbonds,
                        },
                        "frag_hash": frag_hash,
                    }

        return {
            "smiles": smiles,
            "mz_bins": mz_bins,
            "intensities": intensities,
            "num_fragments": len(fragments_result),
            "fragments": fragments_for_plotting,
        }


class EIMSPredictorFromFullEnumeration(_BaseEIMSPredictor):
    """EIMS predictor using full BFS fragment enumeration.

    Two execution paths depending on call site:

    **CPU path** (``predict_from_smiles`` / ``batch_predict_from_smiles``)
        Used for single-molecule or small-batch inference (e.g. interactive notebook).
        ``_generate_fragments`` runs ``FragmentEngine`` — a pure Python/RDKit BFS.
        ``_fragments_to_intensity_inputs`` featurizes each fragment sequentially.

    **GPU path** (``stream_predict_from_smiles`` → ``enumerate_fragments_gpu_batched``)
        Used for large-scale batch inference (e.g. batch_infer.py over PubChem).
        CPU DataLoader workers build only the root DGL graph per molecule.
        ``enumerate_fragments_gpu_batched`` runs the full BFS on CUDA, expanding all
        frontier fragments of all molecules in the batch simultaneously via
        ``batch_remove_single_atoms``. One GPU kernel call per BFS depth level
        instead of ``N_molecules × N_fragments × depth`` Python iterations.
    """

    def __init__(
        self,
        min_mz: float,
        max_mz: float,
        bin_width: float,
        intensity_predictor: Optional[IntensityPredictor] = None,
        intensity_predictor_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            min_mz=min_mz,
            max_mz=max_mz,
            bin_width=bin_width,
            intensity_predictor=intensity_predictor,
            intensity_predictor_checkpoint=intensity_predictor_checkpoint,
            **kwargs,
        )

    def _generate_fragments(
        self,
        smiles: str,
        device: str,
        max_tree_depth: int = 3,
        max_broken_bonds: int = 6,
        num_h_shifts: int = 1,
        max_nodes: Optional[int] = None,
        **kwargs,
    ) -> Optional[FragmentEngine]:
        """CPU path only.

        Not called by stream_predict_from_smiles (GPU path).
        """
        fragment_engine = FragmentEngine(
            mol_str=smiles,
            params=FragmentationParams(
                max_tree_depth=max_tree_depth,
                max_broken_bonds=max_broken_bonds,
                num_h_shifts=num_h_shifts,
                detect_isotope_patterns=False,
                min_isotope_intensity=0.01,
            ),
        )
        # Pure Python/RDKit BFS: removes atoms one at a time, sequential, single-molecule.
        fragment_engine.generate_fragments()

        if (
            max_nodes is not None
            and len(fragment_engine.frag_to_entry) > max_nodes
        ):
            # Keep shallowest / fewest-broken-bond fragments when capping.
            top_entries = sorted(
                fragment_engine.frag_to_entry.items(),
                key=lambda kv: (kv[1].tree_depth, kv[1].max_broken),
            )[:max_nodes]
            fragment_engine.frag_to_entry = dict(top_entries)

        return fragment_engine

    def _fragments_to_intensity_inputs(
        self, root_smiles: str, fragment_engine: FragmentEngine, device: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """CPU path only: featurize FragmentEngine entries into intensity predictor inputs."""
        return self._build_intensity_inputs(
            root_smiles=root_smiles,
            engine=fragment_engine,
            device=device,
            frag_entries_iter=fragment_engine.frag_to_entry.values(),
            get_frag_id=lambda e: e.frag,
            get_max_broken=lambda e: e.max_broken,
            get_base_mass=lambda e: e.base_mass,
        )

    def _gpu_enumerate_fragments(self, root_graph, device, **kwargs):
        """Single-molecule GPU BFS.

        Used for OOM fallback and kernel warmup.
        """
        kwargs.pop("threshold", None)
        return enumerate_fragments_gpu(
            root_graph=root_graph,
            device=device,
            tree_processor=self.tree_processor,
            add_hs=self.intensity_predictor.add_hs,
            embed_elem_group=self.intensity_predictor.embed_elem_group,
            h_shift_range=self.intensity_predictor.h_shift_range,
            encode_formulae=self.intensity_predictor.encode_formulae,
            **kwargs,
        )

    def _gpu_enumerate_fragments_batched(self, root_graphs, device, **kwargs):
        """Batched GPU BFS for a full batch of molecules."""
        kwargs.pop("threshold", None)
        return enumerate_fragments_gpu_batched(
            root_graphs=root_graphs,
            device=device,
            tree_processor=self.tree_processor,
            add_hs=self.intensity_predictor.add_hs,
            embed_elem_group=self.intensity_predictor.embed_elem_group,
            h_shift_range=self.intensity_predictor.h_shift_range,
            encode_formulae=self.intensity_predictor.encode_formulae,
            **kwargs,
        )

    def batch_predict_from_smiles(
        self,
        smiles_list: List[str],
        device: str = "cpu",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Single batched GPU forward pass over a list of SMILES."""
        self.intensity_predictor.to(device)
        self.intensity_predictor.eval()
        self._check_and_fix_dimension_mismatch(device)

        per_mol: List[Optional[tuple]] = []
        for smiles in smiles_list:
            engine = self._generate_fragments(smiles, device, **kwargs)
            if engine is None:
                per_mol.append(None)
                continue
            inputs = self._fragments_to_intensity_inputs(
                smiles, engine, device
            )
            per_mol.append(
                (smiles, engine, inputs) if inputs is not None else None
            )

        valid_idx = [i for i, d in enumerate(per_mol) if d is not None]
        if not valid_idx:
            return [self._empty_spectrum_result() for _ in smiles_list]

        valid = [per_mol[i] for i in valid_idx]
        N = len(valid)
        num_frags_list = [int(d[2]["num_frags"][0].item()) for d in valid]
        max_frags = max(num_frags_list)
        num_frags = torch.tensor(
            num_frags_list, device=device, dtype=torch.long
        )

        all_frag_graphs = dgl.batch([d[2]["graphs"] for d in valid])
        all_root_graphs = dgl.batch([d[2]["root_reprs"] for d in valid])
        ind_maps = torch.cat(
            [
                torch.full((n,), i, dtype=torch.long, device=device)
                for i, n in enumerate(num_frags_list)
            ]
        )

        h_dim = valid[0][2]["masses"].shape[-1]
        broken = torch.zeros(N, max_frags, device=device)
        masses = torch.zeros(N, max_frags, 1, h_dim, device=device)
        max_add = torch.zeros(N, max_frags, device=device)
        max_remove = torch.zeros(N, max_frags, device=device)
        for i, (n, d) in enumerate(zip(num_frags_list, valid)):
            inp = d[2]
            broken[i, :n] = inp["broken_bonds"][0, :n]
            masses[i, :n] = inp["masses"][0, :n]
            max_add[i, :n] = inp["max_add_hs"][0, :n]
            max_remove[i, :n] = inp["max_remove_hs"][0, :n]

        batched_inputs: Dict[str, Any] = {
            "graphs": all_frag_graphs,
            "root_reprs": all_root_graphs,
            "ind_maps": ind_maps,
            "num_frags": num_frags,
            "broken_bonds": broken,
            "masses": masses,
            "max_add_hs": max_add,
            "max_remove_hs": max_remove,
        }

        if valid[0][2].get("root_form_vecs") is not None:
            batched_inputs["root_form_vecs"] = torch.cat(
                [d[2]["root_form_vecs"] for d in valid], dim=0
            )
            formula_dim = valid[0][2]["frag_form_vecs"].shape[-1]
            frag_forms = torch.zeros(N, max_frags, formula_dim, device=device)
            for i, (n, d) in enumerate(zip(num_frags_list, valid)):
                frag_forms[i, :n] = d[2]["frag_form_vecs"][0, :n]
            batched_inputs["frag_form_vecs"] = frag_forms

        with torch.no_grad():
            intensity_result = self.intensity_predictor.predict_intensities(
                graphs=batched_inputs["graphs"],
                root_reprs=batched_inputs["root_reprs"],
                ind_maps=batched_inputs["ind_maps"],
                num_frags=batched_inputs["num_frags"],
                broken_bonds=batched_inputs["broken_bonds"],
                masses=batched_inputs["masses"],
                max_add_hs=batched_inputs.get("max_add_hs"),
                max_remove_hs=batched_inputs.get("max_remove_hs"),
                root_form_vecs=batched_inputs.get("root_form_vecs"),
                frag_form_vecs=batched_inputs.get("frag_form_vecs"),
            )

        results = [self._empty_spectrum_result() for _ in smiles_list]
        for out_i, orig_i in enumerate(valid_idx):
            smiles, engine, _ = valid[out_i]
            results[orig_i] = self._format_spectrum_result(
                {"spec": [intensity_result["spec"][out_i]]}, engine, smiles
            )
        return results

    def _run_batch_oom_fallback(
        self,
        valid_items,
        valid_mol_idx,
        device,
        populate_fragments,
        h_shift_range,
        empty,
        batch_idx_range=None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Per-molecule fallback when batched GPU forward pass hits OOM."""
        per_mol_results: Dict[int, Any] = {}
        for _i, _mol_i in enumerate(valid_mol_idx):
            if _i % 16 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            item = valid_items[_mol_i]
            try:
                inp = self._gpu_enumerate_fragments(
                    item["root_graph"], device, **kwargs
                )
            except Exception as e:
                logging.warning(
                    f"Fragment enumeration failed for {item['smiles']!r}: {e} — skipping."
                )
                torch.cuda.empty_cache()
                continue
            if inp is None:
                continue
            nf = int(inp["num_frags"].item())
            try:
                with torch.no_grad():
                    res = self.intensity_predictor.predict_intensities(
                        graphs=inp["graphs"],
                        root_reprs=inp["root_reprs"],
                        ind_maps=torch.zeros(
                            nf, dtype=torch.long, device=device
                        ),
                        num_frags=inp["num_frags"],
                        broken_bonds=inp["broken_bonds"],
                        masses=inp["masses"],
                        max_add_hs=inp.get("max_add_hs"),
                        max_remove_hs=inp.get("max_remove_hs"),
                        root_form_vecs=inp.get("root_form_vecs"),
                        frag_form_vecs=inp.get("frag_form_vecs"),
                    )
            except Exception as e:
                logging.warning(
                    f"Intensity prediction failed for {item['smiles']!r}: {e} — skipping."
                )
                torch.cuda.empty_cache()
                continue
            frag_masses = (
                inp["masses"][0, :nf, 0, h_shift_range].detach().cpu().numpy()
            )
            if populate_fragments:
                num_nodes = inp["graphs"].batch_num_nodes().cpu()
                n_id = inp["graphs"].ndata["n_id"].cpu()
                ind_maps_cpu = inp["ind_maps"].cpu()
                node_starts = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long),
                        num_nodes[:-1].cumsum(0),
                    ]
                )
                populate_frag_to_entry_from_gpu(
                    ind_maps_cpu=ind_maps_cpu,
                    num_nodes_cpu=num_nodes,
                    node_starts_cpu=node_starts,
                    n_id_all_cpu=n_id,
                    mol_rank=0,
                    base_masses_np=frag_masses,
                    engine=item["engine"],
                )
            del inp
            r = self._format_spectrum_result(
                {"spec": [res["spec"][0]]}, item["engine"], item["smiles"]
            )
            r["num_fragments"] = nf
            r["fragment_masses"] = frag_masses
            per_mol_results[item["idx"]] = r
            del res
        if batch_idx_range is None:
            return [
                per_mol_results.get(item["idx"], empty) for item in valid_items
            ]
        lo, hi = batch_idx_range
        return [per_mol_results.get(i, empty) for i in range(lo, hi + 1)]

    def stream_predict_from_smiles(
        self,
        smiles_list: List[str],
        device: str = "cpu",
        batch_size: int = 64,
        num_workers: int = 0,
        populate_fragments: bool = False,
        failure_log_path: Optional[str] = None,
        **kwargs,
    ):
        """Streaming inference with GPU-accelerated fragment enumeration.

        CPU workers build root DGL graphs; all BFS enumeration runs on device.
        Yields one list of result dicts per batch, in input order.

        Parameters
        ----------
        populate_fragments : bool
            If True, populate result["fragments"] for visualisation. Fragment atom
            membership is recovered from n_id ndata without a separate CPU BFS.
            Set False (default) for large-scale inference where this overhead is unwanted.
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        import torch.multiprocessing as _mp

        _mp.set_sharing_strategy("file_system")

        self.intensity_predictor.to(device)
        self.intensity_predictor.eval()
        self._check_and_fix_dimension_mismatch(device)

        # Pre-warm torch_scatter CUDA kernels to avoid a 2-5 min stall on first batch.
        logging.info("Warming up GPU kernels (torch_scatter JIT)...")
        self._gpu_enumerate_fragments(
            RootGraphInferenceDataset(["CCO"], self)[0]["root_graph"], device
        )
        logging.info("GPU warm-up done.")

        loader = torch.utils.data.DataLoader(
            RootGraphInferenceDataset(
                smiles_list, self, failure_log_path=failure_log_path
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=root_only_collate_fn,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            worker_init_fn=_dataloader_worker_init
            if num_workers > 0
            else None,
            # Workers must not fork a process with an already-initialized CUDA
            # context (the parent has loaded the model onto the GPU) — forking
            # in that state deadlocks. Force spawn so each worker starts clean.
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

        empty = self._empty_spectrum_result()
        h_shift_range = self.intensity_predictor.h_shift_range

        for batch in loader:
            batch_idx_range = batch["batch_idx_range"]
            if not batch["valid"]:
                lo, hi = batch_idx_range
                yield [empty] * (hi - lo + 1)
                continue

            valid_items = batch["valid"]
            valid_mol_idx = list(range(len(valid_items)))
            root_graphs = [item["root_graph"] for item in valid_items]

            try:
                bi_result = self._gpu_enumerate_fragments_batched(
                    root_graphs, device, **kwargs
                )
            except Exception as oom_e:
                is_oom = (
                    isinstance(oom_e, torch.cuda.OutOfMemoryError)
                    or "out of memory" in str(oom_e).lower()
                )
                if not is_oom:
                    logging.warning(
                        f"Fragment enumeration failed on batch of {len(valid_mol_idx)} molecules "
                        f"({type(oom_e).__name__}: {oom_e}) — retrying per-molecule."
                    )
                else:
                    logging.warning(
                        f"CUDA OOM during fragment enumeration on batch of "
                        f"{len(valid_mol_idx)} molecules — retrying per-molecule."
                    )
                del oom_e
                gc.collect()
                torch.cuda.empty_cache()
                yield self._run_batch_oom_fallback(
                    valid_items,
                    valid_mol_idx,
                    device,
                    populate_fragments,
                    h_shift_range,
                    empty,
                    batch_idx_range=batch_idx_range,
                    **kwargs,
                )
                continue

            if bi_result is None:
                lo, hi = batch_idx_range
                yield [empty] * (hi - lo + 1)
                continue

            bi, num_frags_list = bi_result
            num_frags_list = [int(x) for x in num_frags_list]

            try:
                with torch.no_grad():
                    intensity_result = (
                        self.intensity_predictor.predict_intensities(
                            graphs=bi["graphs"],
                            root_reprs=bi["root_reprs"],
                            ind_maps=bi["ind_maps"],
                            num_frags=bi["num_frags"],
                            broken_bonds=bi["broken_bonds"],
                            masses=bi["masses"],
                            max_add_hs=bi.get("max_add_hs"),
                            max_remove_hs=bi.get("max_remove_hs"),
                            root_form_vecs=bi.get("root_form_vecs"),
                            frag_form_vecs=bi.get("frag_form_vecs"),
                        )
                    )
            except Exception as batch_e:
                # Catch broadly (not just OOM-like messages) and fall back
                # to per-molecule retry, which itself catches Exception
                # broadly and skips only the offending molecule (see
                # _run_batch_oom_fallback below). A prior, narrower except
                # clause here only matched OOM/CUDA-error-like substrings
                # and RE-RAISED anything else -- including deterministic,
                # non-OOM errors like a disconnected-component SMILES (e.g.
                # a hydrate/salt such as "O.O.[Na+].[S-2]") producing a
                # zero-edge graph, which crashes deep in the GNN encoder
                # with a torch RuntimeError unrelated to memory
                # ("min(): Expected reduction dim to be specified for
                # input.numel() == 0"). Re-raising there crashed the whole
                # worker process -- and since ~6.8% of PubChem SMILES
                # contain a "." (disconnected components), this was a
                # frequent, deterministic crash source in production,
                # which in turn is what triggered the checkpoint/resume
                # race condition fixed in batch_infer.py so often.
                is_oom = (
                    isinstance(batch_e, torch.cuda.OutOfMemoryError)
                    or "out of memory" in str(batch_e).lower()
                    or "index out of bounds" in str(batch_e).lower()
                    or "CUDA error" in str(batch_e)
                )
                if is_oom:
                    logging.warning(
                        f"CUDA OOM on batch of {len(valid_mol_idx)} molecules — retrying per-molecule."
                    )
                else:
                    logging.warning(
                        f"Intensity prediction failed on batch of {len(valid_mol_idx)} molecules "
                        f"({type(batch_e).__name__}: {batch_e}) — retrying per-molecule."
                    )
                del batch_e
                gc.collect()
                torch.cuda.empty_cache()
                bi = None  # allow GC before fallback
                gc.collect()
                yield self._run_batch_oom_fallback(
                    valid_items,
                    valid_mol_idx,
                    device,
                    populate_fragments,
                    h_shift_range,
                    empty,
                    batch_idx_range=batch_idx_range,
                    **kwargs,
                )
                continue

            # Zero-copy view into masses; .cpu().numpy() happens per-molecule below.
            _base_masses = bi["masses"][:, :, 0, h_shift_range].detach()

            if populate_fragments:
                pop_ind_maps = bi["ind_maps"].cpu()
                pop_num_nodes = bi["graphs"].batch_num_nodes().cpu()
                pop_n_id = bi["graphs"].ndata["n_id"].cpu()
                pop_node_starts = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long),
                        pop_num_nodes[:-1].cumsum(0),
                    ]
                )

            batch_results: Dict[int, Any] = {}
            for rank, mol_i in enumerate(valid_mol_idx):
                item = valid_items[mol_i]
                if populate_fragments:
                    n_f_pop = num_frags_list[rank]
                    populate_frag_to_entry_from_gpu(
                        ind_maps_cpu=pop_ind_maps,
                        num_nodes_cpu=pop_num_nodes,
                        node_starts_cpu=pop_node_starts,
                        n_id_all_cpu=pop_n_id,
                        mol_rank=rank,
                        base_masses_np=_base_masses[rank, :n_f_pop]
                        .cpu()
                        .numpy(),
                        engine=item["engine"],
                    )
                r = self._format_spectrum_result(
                    {"spec": [intensity_result["spec"][rank]]},
                    item["engine"],
                    item["smiles"],
                    num_fragments=num_frags_list[rank],
                )
                n_f = num_frags_list[rank]
                r["num_fragments"] = n_f
                r["fragment_masses"] = _base_masses[rank, :n_f].cpu().numpy()
                batch_results[item["idx"]] = r

            lo, hi = batch_idx_range
            yield [batch_results.get(i, empty) for i in range(lo, hi + 1)]

    def predict_from_smiles_parallel(
        self,
        smiles_list: Union[List[str], str],
        device: str = "cpu",
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Collect all stream_predict_from_smiles batches into a single list.

        Also accepts a path to a CSV file (one SMILES per line) as smiles_list.
        """
        if isinstance(smiles_list, str):
            path = smiles_list
            with open(path) as f:
                smiles_list = [row[0].strip() for row in csv.reader(f) if row]
            logging.info(f"Loaded {len(smiles_list)} SMILES from {path}")

        results = []
        with tqdm(
            total=len(smiles_list), desc="Predicting spectra", unit="mol"
        ) as pbar:
            for batch in self.stream_predict_from_smiles(
                smiles_list,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                populate_fragments=True,
                **kwargs,
            ):
                results.extend(batch)
                pbar.update(len(batch))
        return results

    def _format_spectrum_result(
        self,
        intensity_result: Dict[str, List[torch.Tensor]],
        fragment_engine: FragmentEngine,
        smiles: str,
        num_fragments: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Format spectrum result for FragmentEngine output.

        In the GPU path frag_to_entry is empty unless populate_fragments=True,
        so fragments dict will be empty. Use predict_from_smiles for per-
        fragment visualization metadata.
        """
        intensities = intensity_result["spec"][0].detach().cpu().numpy()
        mz_bins = self.intensity_predictor.inten_buckets.cpu().numpy()

        fragments_for_plotting = {}
        dag_nodes = {}
        if fragment_engine and fragment_engine.frag_to_entry:
            for frag_hash, frag_entry in fragment_engine.frag_to_entry.items():
                if (
                    frag_entry.frag is not None
                    and frag_entry.base_mass is not None
                    and frag_entry.form
                ):
                    draw_info = fragment_engine.get_draw_dict(frag_entry.frag)
                    fragments_for_plotting[float(frag_entry.base_mass)] = {
                        "structure": draw_info.mol,
                        "form": frag_entry.form,
                        "highlights": {
                            "atoms": draw_info.hatoms,
                            "bonds": draw_info.hbonds,
                        },
                        "frag_hash": frag_hash,
                    }
                    dag_nodes[frag_hash] = {
                        "id": frag_entry.id,
                        "base_mass": float(frag_entry.base_mass),
                        "form": frag_entry.form,
                        "tree_depth": frag_entry.tree_depth,
                        "parent_hashes": list(frag_entry.parent_hashes),
                        "structure": draw_info.mol,
                        "highlights": {
                            "atoms": draw_info.hatoms,
                            "bonds": draw_info.hbonds,
                        },
                    }

        return {
            "smiles": smiles,
            "mz_bins": mz_bins,
            "intensities": intensities,
            "num_fragments": num_fragments
            if num_fragments is not None
            else len(fragment_engine.frag_to_entry),
            "fragments": fragments_for_plotting,
            "dag_nodes": dag_nodes,
        }
