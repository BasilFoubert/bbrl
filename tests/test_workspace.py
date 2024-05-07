from typing import Generator
import pytest
import torch
from bbrl.workspace import Workspace


def test_getitem_with_string():
    ws = Workspace()
    ws.set_full("a", torch.tensor([1, 2, 3]))
    result = ws["a"]

    assert isinstance(result, torch.Tensor), "Expected a torch.Tensor"
    assert torch.equal(
        result, torch.tensor([1, 2, 3])
    ), "Tensor does not match expected value"


def test_getitem_with_list_of_strings():
    ws = Workspace()
    ws.set_full("a", torch.tensor([1, 2, 3]))
    ws.set_full("b", torch.tensor([4, 5, 6]))
    result = ws[["a", "b"]]

    # Checking if result is a generator
    assert isinstance(result, Generator), "Expected a generator"

    # Converting generator to a list for easy testing
    result_list = list(result)

    # Checking all items in the list are tensors and match expected values
    assert all(
        isinstance(t, torch.Tensor) for t in result_list
    ), "All items should be torch.Tensor"
    assert torch.equal(
        result_list[0], torch.tensor([1, 2, 3])
    ), "First tensor does not match"
    assert torch.equal(
        result_list[1], torch.tensor([4, 5, 6])
    ), "Second tensor does not match"
