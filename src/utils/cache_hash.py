import ast
import hashlib
import inspect


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return tree


def hash_source(obj):
    """AST-based hash of a function's (or file's) source - ignores whitespace/comments/
    formatting/docstrings, so cosmetic edits don't invalidate caches keyed on this; only real
    code changes do. Pass a function/callable, or a file path string."""
    if callable(obj):
        source = inspect.getsource(obj)
    else:
        with open(obj, encoding="utf-8") as f:
            source = f.read()

    tree = ast.parse(source)
    tree = _strip_docstrings(tree)
    dump = ast.dump(tree, annotate_fields=False)

    return hashlib.sha1(dump.encode()).hexdigest()[:8]


def group_hash(*deps):
    """Combined hash of a group's dependencies - functions (hashed by AST source) and/or plain
    values like module-level constants (hashed by repr, since a constant's value isn't part of
    any function's own source text). A change to ANY dependency invalidates the group's cache."""
    parts = [hash_source(d) if callable(d) else repr(d) for d in deps]
    combined = "".join(parts)
    return hashlib.sha1(combined.encode()).hexdigest()[:8]
