"""Isotope Distribution Calculator."""

import logging
import math
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from icicle.utils import ISOTOPE_PATTERNS, safe_binom


class IsotopePatternCalculator:
    """Calculate isotope patterns for molecular formulas.

    Deprecated for MAGMa labeling / not used for isotope module.
    """

    def __init__(self, min_intensity_threshold: float = 0.01):
        self.min_threshold = min_intensity_threshold

    def parse_formula(self, formula: str) -> Dict[str, int]:
        """Parse molecular formula string into element counts."""
        formula_dict = {}
        i = 0
        while i < len(formula):
            element = formula[i]
            i += 1
            if i < len(formula) and formula[i].islower():
                element += formula[i]
                i += 1

            count_str = ""
            while i < len(formula) and formula[i].isdigit():
                count_str += formula[i]
                i += 1

            count = int(count_str) if count_str else 1
            formula_dict[element] = formula_dict.get(element, 0) + count

        return formula_dict

    def calculate_isotope_distribution(
        self, formula_dict: Dict[str, int]
    ) -> Dict[int, float]:
        """Calculate isotope distribution."""
        distribution = {0: 1.0}  # Start with monoisotopic peak

        for element, count in formula_dict.items():
            if element not in ISOTOPE_PATTERNS or count == 0:
                continue

            isotopes = ISOTOPE_PATTERNS[element]
            if len(isotopes) <= 1:
                continue

            new_distribution = defaultdict(float)

            # FIXED: Handle 2-isotope and multi-isotope elements separately
            if len(isotopes) == 2:
                # Use original binomial method for 2-isotope elements (works perfectly)
                new_distribution = self._calculate_binomial_distribution(
                    distribution, isotopes, count
                )
            else:
                # Use proper multi-isotope method for elements with >2 isotopes
                new_distribution = self._calculate_multi_isotope_distribution(
                    distribution, isotopes, count
                )

            distribution = dict(new_distribution)

        # Normalize to most abundant peak = 1.0
        return self._normalize_distribution(distribution)

    def _calculate_binomial_distribution(
        self, distribution: Dict[int, float], isotopes: list, count: int
    ) -> defaultdict:
        """Original binomial method for 2-isotope elements."""
        new_distribution = defaultdict(float)

        for base_shift, base_intensity in distribution.items():
            max_heavy = min(count, 3)  # Limit to avoid explosion

            for n_heavy in range(max_heavy + 1):
                n_light = count - n_heavy
                if n_light < 0:
                    continue

                heavy_mass_shift = isotopes[1][0]
                total_shift = base_shift + n_heavy * heavy_mass_shift

                # Binomial probability - your original code
                if count <= 10:
                    from math import comb

                    prob = (
                        comb(count, n_heavy)
                        * (isotopes[0][1] ** n_light)
                        * (isotopes[1][1] ** n_heavy)
                    )
                else:
                    lambda_val = count * isotopes[1][1]
                    prob = (
                        np.exp(-lambda_val)
                        * (lambda_val**n_heavy)
                        / math.factorial(n_heavy)
                    )

                intensity = base_intensity * prob
                if intensity >= self.min_threshold:
                    new_distribution[total_shift] += intensity

        return new_distribution

    def _calculate_multi_isotope_distribution(
        self, distribution: Dict[int, float], isotopes: list, count: int
    ) -> defaultdict:
        """Proper multi-isotope method using individual isotope abundances."""
        new_distribution = defaultdict(float)

        for base_shift, base_intensity in distribution.items():
            if count == 1:
                # For single atoms, just use isotope abundances directly
                for iso_shift, abundance in isotopes:
                    if abundance >= 0.001:  # Skip very rare isotopes
                        total_shift = base_shift + iso_shift
                        intensity = base_intensity * abundance
                        if intensity >= self.min_threshold:
                            new_distribution[total_shift] += intensity

            elif count == 2:
                # For pairs of atoms, calculate all combinations
                for i, (shift1, abund1) in enumerate(isotopes):
                    for j, (shift2, abund2) in enumerate(isotopes):
                        if (
                            abund1 * abund2 < 0.00001
                        ):  # Skip very rare combinations
                            continue

                        total_shift = base_shift + shift1 + shift2

                        # Probability: 2 * p1 * p2 if different isotopes, p1^2 if same
                        if i == j:
                            prob = abund1 * abund2
                        else:
                            prob = 2 * abund1 * abund2

                        intensity = base_intensity * prob
                        if intensity >= self.min_threshold:
                            new_distribution[total_shift] += intensity

            else:
                # For count > 2, use approximation with dominant isotope + perturbations
                new_distribution.update(
                    self._approximate_multi_isotope(
                        base_shift, base_intensity, isotopes, count
                    )
                )

        return new_distribution

    def _approximate_multi_isotope(
        self,
        base_shift: int,
        base_intensity: float,
        isotopes: list,
        count: int,
    ) -> Dict[int, float]:
        """Approximate multi-isotope distribution for count > 2."""
        result = {}

        # Find the most abundant isotope
        dominant_idx = max(range(len(isotopes)), key=lambda i: isotopes[i][1])
        dominant_shift, dominant_abundance = isotopes[dominant_idx]

        # Main peak: all atoms are the dominant isotope
        main_shift = base_shift + count * dominant_shift
        main_intensity = base_intensity * (dominant_abundance**count)
        if main_intensity >= self.min_threshold:
            result[main_shift] = main_intensity

        # Add contributions from substituting one dominant isotope with others
        for i, (iso_shift, abundance) in enumerate(isotopes):
            if i == dominant_idx or abundance < 0.01:
                continue

            # Probability of having exactly one of this isotope
            substitution_prob = (
                count * abundance * (dominant_abundance ** (count - 1))
            )
            substitution_shift = (
                base_shift + (count - 1) * dominant_shift + iso_shift
            )
            substitution_intensity = base_intensity * substitution_prob

            if substitution_intensity >= self.min_threshold:
                result[substitution_shift] = (
                    result.get(substitution_shift, 0) + substitution_intensity
                )

        return result

    def _normalize_distribution(
        self, distribution: Dict[int, float]
    ) -> Dict[int, float]:
        """Normalize distribution properly."""
        if not distribution:
            return distribution

        # Find the reference peak for normalization
        # Priority: m+0 peak, then most abundant peak
        if 0 in distribution:
            norm_factor = distribution[0]
        else:
            norm_factor = max(distribution.values())

        if norm_factor > 0:
            return {
                shift: intensity / norm_factor
                for shift, intensity in distribution.items()
                if intensity >= self.min_threshold
            }

        return distribution


class DifferentiableIsotopePatternModule(nn.Module):
    def __init__(
        self,
        element_to_idx: Dict[str, int],
        iso_shifts_data: torch.Tensor,
        iso_abundances_data: torch.Tensor,
        min_intensity_threshold: float = 0.01,
        max_total_isotope_shift: int = 15,
        max_isotope_variants_to_track: int = 5,
    ):
        super().__init__()
        self.element_to_idx = element_to_idx
        self.min_threshold = min_intensity_threshold

        self.register_buffer("iso_shifts_data", iso_shifts_data)
        self.register_buffer("iso_abundances_data", iso_abundances_data)

        self.num_elements = iso_shifts_data.shape[0]
        self.max_elem_iso_variants = iso_shifts_data.shape[1]

        self.max_total_isotope_shift = max_total_isotope_shift
        self.max_isotope_variants_to_track = max_isotope_variants_to_track

    def forward(
        self,
        raw_intensities_per_h_shift: torch.Tensor,  # Shape (N_orig_peaks,)
        base_masses_per_h_shift: torch.Tensor,  # Shape (N_orig_peaks,)
        frag_form_vecs: torch.Tensor,  # Shape (N_orig_peaks, N_elements)
        attn_weights_norm: torch.Tensor,  # Shape (N_orig_peaks,)
        original_batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = raw_intensities_per_h_shift.device
        N_orig_peaks = raw_intensities_per_h_shift.shape[0]

        # Filter out peaks with very low base intensity early
        non_zero_base_mask = (
            raw_intensities_per_h_shift * attn_weights_norm > 1e-9
        )

        if not non_zero_base_mask.any():
            return (
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device),
            )

        # Apply mask
        base_peak_intensities = (
            raw_intensities_per_h_shift * attn_weights_norm
        )[non_zero_base_mask]
        base_masses_per_h_shift = base_masses_per_h_shift[non_zero_base_mask]
        frag_form_vecs = frag_form_vecs[non_zero_base_mask]
        # MASK THE NEW INPUT TOO
        active_original_batch_indices = original_batch_indices[
            non_zero_base_mask
        ]

        N_active_peaks = base_peak_intensities.shape[0]

        if not non_zero_base_mask.any():
            return (
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device),
            )

        current_iso_dist_probs = torch.zeros(
            N_active_peaks,
            self.max_total_isotope_shift + 1,
            device=device,
            dtype=torch.float32,
        )
        current_iso_dist_probs[:, 0] = 1.0

        for elem_idx in range(self.num_elements):
            elem_counts = frag_form_vecs[:, elem_idx].long()

            if torch.all(elem_counts == 0):
                continue

            elem_iso_shifts_raw = self.iso_shifts_data[elem_idx, :]
            elem_iso_abundances_raw = self.iso_abundances_data[elem_idx, :]

            valid_elem_iso_mask = elem_iso_abundances_raw > 0.01
            elem_iso_shifts = elem_iso_shifts_raw[valid_elem_iso_mask]
            elem_iso_abundances = elem_iso_abundances_raw[valid_elem_iso_mask]

            if len(elem_iso_shifts) <= 1:
                continue

            # Calculate contribution of `elem_counts` atoms of this element

            # Max k for current element's heavy isotopes (for binomial-like expansion)
            # Cap max_k_for_elem at the maximum count present, and also at max_isotope_variants_to_track
            # for performance/memory.
            max_k_for_elem_local = min(
                elem_counts.max().item(), self.max_isotope_variants_to_track
            )
            k_values = torch.arange(
                max_k_for_elem_local + 1, device=device
            ).float()

            elem_counts_expanded = elem_counts.unsqueeze(
                -1
            ).float()  # (N_active_peaks, 1)

            # Initialize probabilities and shifts for this element's contribution
            elem_contrib_probs = torch.zeros(
                N_active_peaks,
                self.max_isotope_variants_to_track,
                device=device,
            )
            elem_contrib_shifts = torch.zeros(
                self.max_isotope_variants_to_track, device=device
            ).float()

            if (
                len(elem_iso_shifts) == 2
            ):  # Binary isotope system (e.g., C, B, Cl, Br, H, N, O)
                p0 = elem_iso_abundances[0]  # Abundance of light isotope
                p1 = elem_iso_abundances[1]  # Abundance of heavy isotope
                shift_unit = elem_iso_shifts[
                    1
                ]  # Mass difference of the heavy isotope

                # Binomial coefficients C(n, k) -> (N_active_peaks, max_k_for_elem_local+1)
                binom_coeffs = safe_binom(elem_counts_expanded, k_values)

                # Probabilities for k heavy isotopes: p_light^(count-k) * p_heavy^k
                elem_iso_probs_per_k = (
                    p0 ** (elem_counts_expanded - k_values)
                ) * (p1**k_values)

                # Total probability contribution for k heavy isotopes for each peak:
                elem_contrib_probs = (
                    binom_coeffs * elem_iso_probs_per_k
                )  # (N_active_peaks, max_k_for_elem_local+1)
                elem_contrib_shifts = (
                    k_values * shift_unit
                )  # (max_k_for_elem_local+1,)

                if (
                    torch.isnan(elem_contrib_probs).any()
                    or torch.isinf(elem_contrib_probs).any()
                ):
                    logging.error(
                        f"ERROR: NaN/Inf in elem_contrib_probs after binomial for element {list(self.element_to_idx.keys())[elem_idx]}!"
                    )
                    raise ValueError(
                        "NaN/Inf detected in isotope probabilities!"
                    )

            else:  # Multi-isotope system (e.g., Si, S, K, Fe, Se after simplification)
                # This is the simplified multi-isotope approximation.
                # For `elem_counts > 1`, we approximate with the dominant isotope and single substitutions.

                dominant_idx = torch.argmax(elem_iso_abundances)
                dominant_shift = elem_iso_shifts[dominant_idx]
                dominant_abundance = elem_iso_abundances[dominant_idx]

                # Main peak (all dominant isotopes): k=0 equivalent for this element's contribution
                elem_contrib_probs[:, 0] = (
                    dominant_abundance ** elem_counts.float()
                )
                elem_contrib_shifts[0] = dominant_shift

                # Single substitutions (approximate):
                substitution_k_idx = 1  # Start filling from the next index in elem_contrib_probs/shifts
                for iso_idx in range(len(elem_iso_shifts)):
                    if iso_idx == dominant_idx:
                        continue
                    if elem_iso_abundances[iso_idx] < self.min_threshold:
                        continue  # Only significant substitutions

                    if substitution_k_idx < self.max_isotope_variants_to_track:
                        # Probability of having exactly one of this substituting isotope
                        sub_prob = (
                            elem_counts.float()
                            * elem_iso_abundances[iso_idx]
                            * (dominant_abundance ** (elem_counts.float() - 1))
                        )

                        elem_contrib_probs[:, substitution_k_idx] = sub_prob
                        elem_contrib_shifts[substitution_k_idx] = (
                            elem_iso_shifts[iso_idx]
                        )
                        substitution_k_idx += 1

                # Normalize probabilities for this element's contribution to 1 (conceptually)
                sum_elem_contrib_probs = elem_contrib_probs.sum(
                    dim=1, keepdim=True
                )
                elem_contrib_probs = torch.where(
                    sum_elem_contrib_probs > 0,
                    elem_contrib_probs / sum_elem_contrib_probs,
                    torch.tensor(0.0, device=device),
                )

                if (
                    torch.isnan(elem_contrib_probs).any()
                    or torch.isinf(elem_contrib_probs).any()
                ):
                    logging.error(
                        f"ERROR: NaN/Inf in elem_contrib_probs after multi-isotope for element {list(self.element_to_idx.keys())[elem_idx]}!"
                    )
                    raise ValueError(
                        "NaN/Inf detected in isotope probabilities!"
                    )

            if (
                torch.isnan(current_iso_dist_probs).any()
                or torch.isinf(current_iso_dist_probs).any()
            ):
                logging.error(
                    "ERROR: NaN/Inf in current_iso_dist_probs before convolution!"
                )
                raise ValueError("NaN/Inf detected in isotope probabilities!")

            # Batched discrete convolution
            new_iso_dist_probs = torch.zeros(
                N_active_peaks, self.max_total_isotope_shift + 1, device=device
            )

            # Expand for element-wise product across shifts
            # (N_active_peaks, current_shifts_len, 1) * (N_active_peaks, 1, elem_shifts_len)
            product_of_probs = current_iso_dist_probs.unsqueeze(
                -1
            ) * elem_contrib_probs.unsqueeze(1)

            if (
                torch.isnan(product_of_probs).any()
                or torch.isinf(product_of_probs).any()
            ):
                logging.error("ERROR: NaN/Inf in product_of_probs!")
                raise ValueError("NaN/Inf detected in isotope probabilities!")

            # Calculate target indices for scattering:
            # (N_active_peaks, current_shifts_len, 1) + (1, 1, elem_shifts_len)
            current_shifts_range = (
                torch.arange(self.max_total_isotope_shift + 1, device=device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )

            # Expand elem_contrib_shifts to match shape for addition
            elem_contrib_shifts_expanded = elem_contrib_shifts.unsqueeze(
                0
            ).unsqueeze(0)  # (1, 1, max_k_for_elem_local+1)

            target_shift_indices = (
                current_shifts_range + elem_contrib_shifts_expanded
            ).long()

            # Clamp target shifts to stay within bounds
            target_shift_indices = target_shift_indices.expand(
                N_active_peaks, -1, -1
            )
            target_shift_indices = torch.clamp(
                target_shift_indices, 0, self.max_total_isotope_shift
            )

            # Flatten for scatter_add_
            flat_target_shift_indices = target_shift_indices.reshape(-1)
            flat_product_of_probs = product_of_probs.reshape(-1)

            # Create corresponding flat peak indices for scatter_add_
            peak_indices_expanded = (
                torch.arange(N_active_peaks, device=device)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand_as(product_of_probs)
                .reshape(-1)
            )

            global_scatter_indices = (
                peak_indices_expanded * (self.max_total_isotope_shift + 1)
                + flat_target_shift_indices
            )

            new_iso_dist_probs.view(-1).scatter_add_(
                0, global_scatter_indices, flat_product_of_probs
            )

            if (
                torch.isnan(new_iso_dist_probs).any()
                or torch.isinf(new_iso_dist_probs).any()
            ):
                logging.error(
                    "ERROR: NaN/Inf in new_iso_dist_probs after scatter_add_!"
                )
                raise ValueError("NaN/Inf detected in isotope probabilities!")

            current_iso_dist_probs = new_iso_dist_probs

        # Normalize each peak's final isotope distribution
        max_probs_per_peak = torch.max(
            current_iso_dist_probs, dim=1, keepdim=True
        ).values
        max_probs_per_peak = torch.where(
            max_probs_per_peak > 0,
            max_probs_per_peak,
            torch.tensor(1e-9, device=device),
        )

        normalized_isotope_patterns = (
            current_iso_dist_probs / max_probs_per_peak
        )

        if (
            torch.isnan(normalized_isotope_patterns).any()
            or torch.isinf(normalized_isotope_patterns).any()
        ):
            logging.error("ERROR: NaN/Inf in normalized_isotope_patterns!")
            raise ValueError("NaN/Inf detected in isotope probabilities!")

        # Apply overall min_intensity_threshold
        normalized_isotope_patterns = torch.where(
            normalized_isotope_patterns >= self.min_threshold,
            normalized_isotope_patterns,
            torch.tensor(0.0, device=device),
        )

        # Combine with original base peak intensities
        final_expanded_intensities = (
            base_peak_intensities.unsqueeze(-1) * normalized_isotope_patterns
        )

        if (
            torch.isnan(final_expanded_intensities).any()
            or torch.isinf(final_expanded_intensities).any()
        ):
            logging.error("ERROR: NaN/Inf in final_expanded_intensities!")
            raise ValueError("NaN/Inf detected in isotope probabilities!")

        # Calculate actual masses for each expanded isotope peak
        mass_shifts_tensor = (
            torch.arange(self.max_total_isotope_shift + 1, device=device)
            .float()
            .unsqueeze(0)
        )
        final_expanded_masses = (
            base_masses_per_h_shift.unsqueeze(-1) + mass_shifts_tensor
        )

        # Filter out zero-intensity peaks
        valid_final_mask = final_expanded_intensities > 1e-9

        all_isotope_masses_t = final_expanded_masses[valid_final_mask]
        all_isotope_intensities_t = final_expanded_intensities[
            valid_final_mask
        ]

        # Use the original batch indices, masked correctly.
        all_isotope_batch_indices_t = active_original_batch_indices.unsqueeze(
            -1
        ).expand_as(final_expanded_intensities)[valid_final_mask]

        return (
            all_isotope_masses_t,
            all_isotope_intensities_t,
            all_isotope_batch_indices_t,
        )
