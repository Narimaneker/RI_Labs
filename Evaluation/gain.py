def gain(eval_sys_a : float, eval_sys_b: float) -> float:
    if eval_sys_b == 0.0:
        return 0.0
    return ((eval_sys_a - eval_sys_b) / eval_sys_b) * 100