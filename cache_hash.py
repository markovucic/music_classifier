import ast
import hashlib
import inspect


def hash_source(obj):
    """AST-based hash of a function's (or file's) source - ignores whitespace/comments/
    formatting, so cosmetic edits don't invalidate caches keyed on this; only real code
    changes do. Pass a function/callable, or a file path string."""
    if callable(obj):
        source = inspect.getsource(obj)
    else:
        with open(obj, encoding="utf-8") as f:
            source = f.read()

    tree = ast.parse(source)
    dump = ast.dump(tree, annotate_fields=False)

    return hashlib.sha1(dump.encode()).hexdigest()[:8]


def group_hash(*deps):
    """Combined hash of a group's dependencies - functions (hashed by AST source) and/or plain
    values like module-level constants (hashed by repr, since a constant's value isn't part of
    any function's own source text). A change to ANY dependency invalidates the group's cache."""
    parts = [hash_source(d) if callable(d) else repr(d) for d in deps]
    combined = "".join(parts)
    return hashlib.sha1(combined.encode()).hexdigest()[:8]
