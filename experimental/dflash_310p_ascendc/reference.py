# SPDX-License-Identifier: Apache-2.0
"""CPU oracle for text-only MRv1 DFlash metadata (no torch dependency)."""

LANES = 8
QUERY_FIELDS = 6
PAD_SLOT = -1


def _slot(position, table, page_size, page_count, null_page, max_model_len, *, is_context=False):
    if position >= max_model_len and not is_context:
        return PAD_SLOT, 0
    if 0 <= position < max_model_len and page_size > 0 and page_count > 0:
        page_index, offset = divmod(position, page_size)
        page = table[page_index] if page_index < len(table) else -1
        if is_context and 0 <= page == null_page:
            return PAD_SLOT, 0
        if 0 <= page < page_count and page != null_page:
            value = page * page_size + offset
            if value <= 2**31 - 1:
                return value, 0
    return PAD_SLOT, 2


def prepare_reference(requests, positions, block_tables, group_info, spec, max_model_len, mask_id, from_anchor):
    all_queries, all_context = [], []
    for tables, (page_size, page_count, null_page, *_) in zip(block_tables, group_info):
        query_rows = []
        context_positions = [0] * len(positions)
        context_slots = [PAD_SLOT] * len(positions)
        for row, ((start, end, rejected, bonus, *_), table) in enumerate(zip(requests, tables)):
            ids, query_positions, query_slots, samples, sample_positions, info = (
                [0] * LANES,
                [0] * LANES,
                [PAD_SLOT] * LANES,
                [0] * LANES,
                [0] * LANES,
                [0] * LANES,
            )
            query_rows.append((ids, query_positions, query_slots, samples, sample_positions, info))
            width = spec + 1
            start_query = row * width
            info[:2] = [start_query, start_query]
            if start == -1:
                continue
            if not (0 <= start < end <= len(positions) and 0 <= rejected <= end - start and bonus >= 0):
                info[3] = 1
                continue
            accepted = positions[start : end - rejected]
            anchor = accepted[-1] if accepted else positions[start]
            if not 0 <= anchor < max_model_len:
                info[3] = 1
                continue
            first_query_pos = anchor + bool(accepted)

            ids[:width] = [bonus] + [mask_id] * spec
            for i in range(width):
                pos = first_query_pos + i
                query_positions[i] = min(pos, max_model_len - 1)
                query_slots[i], status = _slot(pos, table, page_size, page_count, null_page, max_model_len)
                info[3] |= status
            for i in range(spec):
                samples[i] = start_query + i + int(not from_anchor)
                sample_positions[i] = first_query_pos + i + 1
            info[1:3] = [start_query + width, min(first_query_pos + width, max_model_len)]
            for i, pos in enumerate(accepted, start):
                context_slots[i], status = _slot(
                    pos, table, page_size, page_count, null_page, max_model_len, is_context=True
                )
                if status == 0:
                    context_positions[i] = pos
                info[3] |= status
        all_queries.append(query_rows)
        all_context.append((context_positions, context_slots))
    return all_queries, all_context


def cases():
    """Same capacities with different live rows for graph replay."""
    info = [[16, 32, 0, 0, 0, 0, 0, 0], [8, 32, 0, 0, 0, 0, 0, 0]]
    tables = [
        [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]],
        [[2, 5, 6, 9], [11, 12, 13, 14], [15, 18, 21, 23]],
    ]
    mixed = (
        [[0, 14, 2, 102, 0, 0, 0, 0], [14, 16, 2, 104, 0, 0, 0, 0], [-1] + [0] * 7],
        list(range(8, 22)) + [28, 29] + [99] * 8,
        tables,
        info,
    )
    reused = (
        [[0, 3, 0, 109, 0, 0, 0, 0], [-1] + [0] * 7, [-1] + [0] * 7],
        [2, 3, 4] + [99] * 21,
        [group[::-1] for group in tables],
        info,
    )
    pruned_tables = [[row.copy() for row in group] for group in tables]
    pruned_tables[1][0][0] = 0
    long_context = (
        [[0, 21, 2, 107, 0, 0, 0, 0], [-1] + [0] * 7, [-1] + [0] * 7],
        list(range(1, 22)) + [99] * 3,
        pruned_tables,
        info,
    )
    invalid_tables = [[row.copy() for row in group] for group in tables]
    invalid_tables[1][0][0] = 50
    invalid = (
        [[0, 3, 0, 109, 0, 0, 0, 0], [3, 30, 0, 104, 0, 0, 0, 0], [-1] + [0] * 7],
        [2, 3, 4] + [99] * 21,
        invalid_tables,
        info,
    )
    return {"mixed": mixed, "long_context": long_context, "reuse": reused, "invalid": invalid}


def large_prefill_cases(rows: int, max_model_len: int):
    """Use the same physical context capacity for a full and a partial tail."""
    capacity = 8192
    page_sizes = (128, 64)
    table_width = (max_model_len + min(page_sizes) - 1) // min(page_sizes)
    tables = [
        [list(range(row * table_width + 1, (row + 1) * table_width + 1)) for row in range(rows)] for _ in page_sizes
    ]
    group_info = [[page_size, rows * table_width + 1, 0, 0, 0, 0, 0, 0] for page_size in page_sizes]
    result = {}
    for active in (capacity, capacity - 1):
        requests, positions = [], []
        tokens_per_row, remainder = divmod(active, rows)
        for row in range(rows):
            count = tokens_per_row + (row < remainder)
            start = len(positions)
            positions.extend(range(row % 3, row % 3 + count))
            requests.append([start, start + count, 0, 107 + row, 0, 0, 0, 0])
        positions.extend([0] * (capacity - active))
        result[f"prefill_{rows}_{active}"] = (requests, positions, tables, group_info)
    return result
