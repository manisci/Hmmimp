import pytest 
def capital(x):
    if not isinstance(x,str):
        raise TypeError("please provide a string ")
    return x.capitalize()
def test_ans():
    assert capital("mani") == "Mani"
def test_raiseexpon_nonstr():
    with pytest.raises(TypeError):
        capital(9)
