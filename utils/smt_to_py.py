import argparse
import os
from fractions import Fraction
from typing import List, Tuple

from z3 import parse_smt2_file, simplify
import z3.z3util as zutil


def parse_affine(expr):
    """Return coefficients dict and constant for a linear arithmetic expression."""
    from z3 import is_rational_value, Z3_OP_UNINTERPRETED

    if is_rational_value(expr):
        return {}, Fraction(expr.as_fraction())
    if expr.num_args() == 0 and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        return {str(expr): Fraction(1)}, Fraction(0)

    name = expr.decl().name()
    if name == '+':
        coeffs = {}
        const = Fraction(0)
        for ch in expr.children():
            c, k = parse_affine(ch)
            for v, val in c.items():
                coeffs[v] = coeffs.get(v, 0) + val
            const += k
        return coeffs, const
    if name == '-':
        if len(expr.children()) == 1:
            c, k = parse_affine(expr.children()[0])
            return {v: -val for v, val in c.items()}, -k
        c1, k1 = parse_affine(expr.children()[0])
        c2, k2 = parse_affine(expr.children()[1])
        coeffs = c1.copy()
        for v, val in c2.items():
            coeffs[v] = coeffs.get(v, 0) - val
        return coeffs, k1 - k2
    if name == '*':
        a, b = expr.children()
        if is_rational_value(a):
            scalar = Fraction(a.as_fraction())
            c, k = parse_affine(b)
        elif is_rational_value(b):
            scalar = Fraction(b.as_fraction())
            c, k = parse_affine(a)
        else:
            raise ValueError("Non-linear multiplication")
        return {v: scalar * val for v, val in c.items()}, scalar * k
    if name == '/':
        a, b = expr.children()
        if is_rational_value(b):
            scalar = Fraction(1, 1) / Fraction(b.as_fraction())
            c, k = parse_affine(a)
            return {v: scalar * val for v, val in c.items()}, scalar * k
        raise ValueError("Non-linear division")
    raise ValueError(f"Unsupported expression: {expr}")


def atom_from_z3(expr, variables: List[str]):
    op = expr.decl().name()
    if expr.num_args() != 2:
        raise ValueError("Unexpected atom format")
    left, right = expr.children()
    if op == '<=':
        coeffs, const = parse_affine(left - right)
        const = -const
    elif op == '>=':
        coeffs, const = parse_affine(right - left)
        const = -const
    elif op == '=':
        coeffs, const = parse_affine(left - right)
        const = -const
        op = '='
    else:
        raise NotImplementedError(f"Operator {op} not supported")

    atom = []
    for idx, var in enumerate(variables):
        val = coeffs.get(var, 0)
        if val != 0:
            atom.append((idx, float(round(float(val), 5))))
    atom.append((op, float(round(float(const), 5))))
    return atom


def parse_smt2(filename: str):
    expr = parse_smt2_file(filename)[0]
    expr = simplify(expr, som=True)
    variables = sorted({str(v) for v in zutil.get_vars(expr)})

    if expr.decl().name() == 'or':
        clauses_z3 = list(expr.children())
    else:
        clauses_z3 = [expr]

    clauses: List[List[List[Tuple[int, float]]]] = []
    for cexpr in clauses_z3:
        if cexpr.decl().name() == 'and':
            atoms_z3 = cexpr.children()
        else:
            atoms_z3 = [cexpr]
        atoms = [atom_from_z3(a, variables) for a in atoms_z3]
        clauses.append(atoms)
    return variables, clauses


def generate_py(name: str, variables: List[str], clauses, outdir: str):
    cnt_reals = len(variables)
    lines = []
    lines.append("#!/usr/bin/env python3")
    lines.append("import sys")
    lines.append("import os")
    lines.append(
        "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
    )
    lines.append("from SimpleWMISolver import SimpleWMISolver")
    lines.append("from utils.realsUniverse import RealsUniverse")
    lines.append("from utils.weightFunction import WeightFunction")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("")
    lines.append("def main():")
    lines.append(f"    cntReals = {cnt_reals}")
    lines.append("    cntBools = 0")
    lines.append(
        "    uni = RealsUniverse(cntReals, lowerBound=-1000, upperBound=1000)"
    )
    lines.append("    expr = [")
    for cl in clauses:
        lines.append("        [")
        for at in cl:
            parts = []
            for t in at:
                if isinstance(t[0], int):
                    parts.append(f"({t[0]}, {t[1]})")
                else:
                    parts.append(f"('{t[0]}', {t[1]})")
            lines.append("            [" + ", ".join(parts) + "],")
        lines.append("        ],")
    lines.append("    ]")
    zeros = ", ".join(["0"] * cnt_reals)
    lines.append(f"    monomials = [[1, [{zeros}]]]" )
    lines.append("    poly_wf = WeightFunction(monomials, np.array([]))")
    lines.append("    eps = 0.25")
    lines.append("    delta = 0.15")
    lines.append("    task = SimpleWMISolver(expr, cntBools, uni, poly_wf)")
    lines.append("    result = task.simpleCoverage(eps, delta)")
    lines.append("    print(result)")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{name}.py")
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SMT2 file to Python benchmark'
    )
    parser.add_argument('smt2_file')
    parser.add_argument('--outdir', default='benchmarks')
    args = parser.parse_args()

    variables, clauses = parse_smt2(args.smt2_file)
    base = os.path.splitext(os.path.basename(args.smt2_file))[0]
    generate_py(base, variables, clauses, args.outdir)


if __name__ == '__main__':
    main()