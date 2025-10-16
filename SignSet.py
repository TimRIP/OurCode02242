
from dataclasses import dataclass
from typing import TypeAlias, Literal
from hypothesis import given
from hypothesis.strategies import integers, sets

# ------------------------------------------------------------
# Basic Sign Abstraction
# ------------------------------------------------------------

Sign: TypeAlias = Literal["+", "-", "0"]

@dataclass(frozen=True)
class SignSet:
    """Abstraction of integer sign sets (+, -, 0)."""
    signs: set[Sign]

    # α: abstraction function  (maps concrete set → SignSet)
    @classmethod
    def abstract(cls, items: set[int]):
        signset: set[Sign] = set()
        if 0 in items:
            signset.add("0")
        if any(x > 0 for x in items):
            signset.add("+")
        if any(x < 0 for x in items):
            signset.add("-")
        return cls(signset)

    # γ: concretization membership  (integer ∈ SignSet?)
    def __contains__(self, member: int) -> bool:
        if member == 0 and "0" in self.signs:
            return True
        if member > 0 and "+" in self.signs:
            return True
        if member < 0 and "-" in self.signs:
            return True
        return False

    # Poset operations: order, join, meet
    def __le__(self, other: "SignSet") -> bool:
        return self.signs.issubset(other.signs)

    def __or__(self, other: "SignSet") -> "SignSet":
        return SignSet(self.signs | other.signs)

    def __and__(self, other: "SignSet") -> "SignSet":
        return SignSet(self.signs & other.signs)

    # Abstract addition in the Sign domain
    def __add__(self, other: "SignSet") -> "SignSet":
        result = set()
        for a in self.signs:
            for b in other.signs:
                # Over-approximation addition table
                if a == "0":
                    result.add(b)
                elif b == "0":
                    result.add(a)
                elif a == "+" and b == "+":
                    result.add("+")
                elif a == "-" and b == "-":
                    result.add("-")
                else:
                    # mixed signs → could be any of -, 0, +
                    result |= {"+", "-", "0"}
        return SignSet(result)

    def __str__(self):
        return f"SignSet({self.signs})"


# ------------------------------------------------------------
# Abstract Arithmetic: Over-approximation of integer ops
# ------------------------------------------------------------

class Arithmetic:
    """Implements arithmetic operations over SignSet."""

    @staticmethod
    def add(a: SignSet, b: SignSet) -> SignSet:
        return a + b

    @staticmethod
    def sub(a: SignSet, b: SignSet) -> SignSet:
        """Subtraction is same as addition but invert the second operand."""
        inverted = set()
        for s in b.signs:
            if s == "+": inverted.add("-")
            elif s == "-": inverted.add("+")
            elif s == "0": inverted.add("0")
        return a + SignSet(inverted)

    @staticmethod
    def mul(a: SignSet, b: SignSet) -> SignSet:
        """Abstract multiplication."""
        result = set()
        for x in a.signs:
            for y in b.signs:
                if "0" in (x, y):
                    result.add("0")
                elif x == y:
                    result.add("+")
                else:
                    result.add("-")
        return SignSet(result)

    @staticmethod
    def compare(op: str, a: SignSet, b: SignSet) -> set[bool]:
        """
        Abstract comparison (e.g., ≤).
        Returns a set of booleans {True, False} possible for the given signs.
        """
        outcomes = set()
        for x in a.signs:
            for y in b.signs:
                match op:
                    case "le":  # less or equal
                        if x == "-" and y == "-":
                            outcomes |= {True, False}
                        elif x == "-" and y in {"0", "+"}:
                            outcomes.add(True)
                        elif x == "0" and y in {"0", "+"}:
                            outcomes.add(True)
                        elif x == "+" and y == "+":
                            outcomes |= {True, False}
                        elif x == "+" and y in {"0", "-"}:
                            outcomes.add(False)
                        elif x == "0" and y == "-":
                            outcomes.add(False)
                        else:
                            outcomes |= {True, False}
                    case _:
                        outcomes |= {True, False}
        return outcomes
# 1 Galois connection test: ∀X⊂2ℤ : X ⊆ γ(α(X))
@given(sets(integers()))
def test_valid_abstraction(xs):
    s = SignSet.abstract(xs)
    assert all(x in s for x in xs)


#  Abstract addition test: α(A + B) ⊆ α(A) +Sign α(B)
@given(sets(integers()), sets(integers()))
def test_sign_adds(xs, ys):
    concrete_sum = {x + y for x in xs for y in ys}
    left = SignSet.abstract(concrete_sum)
    right = Arithmetic.add(SignSet.abstract(xs), SignSet.abstract(ys))
    assert left <= right


# 3️ Abstract comparison test (≤)
@given(sets(integers()), sets(integers()))
def test_sign_compare_le(xs, ys):
    concrete_cmp = {x <= y for x in xs for y in ys}
    abstract_cmp = Arithmetic.compare("le", SignSet.abstract(xs), SignSet.abstract(ys))
    assert concrete_cmp.issubset(abstract_cmp)

if __name__ == "__main__":
    print("ok")
