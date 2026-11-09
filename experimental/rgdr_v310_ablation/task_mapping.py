"""CPU model of the direct (request, value-head) task partition."""


def assigned_tasks(lengths: list[int], value_heads: int, cores: int) -> list[list[tuple[int, int, int]]]:
    if value_heads <= 0 or cores <= 0 or any(length < 0 for length in lengths):
        raise ValueError("invalid task geometry")
    total = len(lengths) * value_heads
    result: list[list[tuple[int, int, int]]] = []
    for core in range(cores):
        first = total * core // cores
        last = total * (core + 1) // cores
        first_batch = first // value_heads
        seq0 = sum(lengths[:first_batch])
        tasks: list[tuple[int, int, int]] = []
        task = first
        while task < last:
            batch = task // value_heads
            head_begin = task - batch * value_heads
            head_end = min(value_heads, last - batch * value_heads)
            task = batch * value_heads + head_end
            length = lengths[batch]
            if length:
                tasks.extend((batch, head, seq0) for head in range(head_begin, head_end))
            seq0 += length
        result.append(tasks)
    return result
