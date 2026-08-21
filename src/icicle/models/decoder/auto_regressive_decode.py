"""Function to auto-regressively decode a spectrum using a fragment engine.

This function is used to decode a spectrum using a fragment engine. It is used
to generate a tree of fragments that can be used to reconstruct the spectrum.
"""


def auto_regressive_decode(
    sorted_order,
    frag_hash_to_entry,
    frag_to_hash,
    form_to_min_score,
    engine,
    min_prob,
    id_,
    depth,
    max_nodes,
    threshold,
):
    new_stack = []
    for new_item in sorted_order:
        prob_gen = new_item["prob_gen"]
        atom_ind = new_item["atom_ind"]
        atom_pred = new_item["atom_pred"]

        # Filter out on minimum prob
        if prob_gen <= min_prob:
            continue

        # Calc stack ind
        orig_entry = new_item["orig_entry"]
        frag_int = orig_entry[0]
        frag_hash = orig_entry[1]
        dgl_new_to_old = orig_entry[3]

        # Get atom ind
        atom = dgl_new_to_old[atom_ind]

        # Calc remove dict
        out_dicts = engine.remove_atom(frag_int, int(atom))

        # Update atoms_pulled for parent
        frag_hash_to_entry[frag_hash]["atoms_pulled"].append(int(atom))
        frag_hash_to_entry[frag_hash]["left_pred"].append(float(atom_pred))
        parent_broken = frag_hash_to_entry[frag_hash]["max_broken"]

        for out_dict in out_dicts:
            out_hash = out_dict.new_hash
            out_frag = out_dict.new_frag
            rm_bond_t = out_dict.rm_bond_t
            frag_to_hash[out_frag] = out_hash
            current_entry = frag_hash_to_entry.get(out_hash)

            max_broken = parent_broken + rm_bond_t

            # Define probability of generating
            if current_entry is None:
                score = engine.score_fragment(int(out_frag))[1]

                new_stack.append(out_frag)
                new_entry = {
                    "frag": int(out_frag),
                    "frag_hash": out_hash,
                    "score": score,
                    "id": id_,
                    "parents": [frag_hash],
                    "atoms_pulled": [],
                    "left_pred": [],
                    "max_broken": max_broken,
                    "tree_depth": depth,
                    "prob_gen": prob_gen,
                }
                id_ += 1
                new_entry.update(
                    engine.atom_pass_stats(out_frag, depth=max_broken)
                )

                # reset to best score
                temp_form = new_entry["form"]
                prev_best_score = form_to_min_score.get(
                    temp_form, float("inf")
                )
                form_to_min_score[temp_form] = min(
                    new_entry["score"], prev_best_score
                )
                frag_hash_to_entry[out_hash] = new_entry

            else:
                current_entry["parents"].append(frag_hash)
                current_entry["prob_gen"] = max(
                    current_entry["prob_gen"], prob_gen
                )

            # Update cur probs for the current batch index
            # This is inefficient and can be made smarter without
            # doing another minimum calculation
            cur_probs = sorted(
                [i["prob_gen"] for i in frag_hash_to_entry.values()]
            )[::-1]
            if max_nodes is None or len(cur_probs) < max_nodes:
                min_prob = threshold
            elif max_nodes is not None and len(cur_probs) >= max_nodes:
                min_prob = cur_probs[max_nodes - 1]
            else:
                raise NotImplementedError()

    return {
        "frag_hash_to_entry": frag_hash_to_entry,
        "frag_to_hash": frag_to_hash,
        "form_to_min_score": form_to_min_score,
        "min_prob": min_prob,
        "id_": id_,
        "sorted_order": sorted_order,
        "new_stack": new_stack,
    }
