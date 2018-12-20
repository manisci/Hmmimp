import pytest 
from wallet import Wallet,Insuffexcep


@pytest.fixture

def mywallet():
    ''' creates empty wallet '''
    wallet = Wallet()
    return wallet
@pytest.fixture
def twentywallet():
    ''' creates wallet of 20 '''
    return Wallet(20)

@pytest.mark.parametrize("init,spent,expect",[(30,10,20),(70,40,30),(30,30,0)])

def test_transaction(mywallet,init,spent,expect):
    mywallet.add_cash(init)
    mywallet.spend_cash(spent)
    assert mywallet.balance == expect

# def test_default_initial_amount(empty_wallet):
#     assert empty_wallet.balance == 0
# def test_setting_initial_amount(twentywallet):
#     assert twentywallet.balance == 20
# def test_walllet_add_cash(empty_wallet):
#     empty_wallet.add_cash(10)
#     assert empty_wallet.balance == 10
# def test_wallet_sepend_cash(twentywallet):
#     twentywallet.spend_cash(10)
#     assert twentywallet.balance == 10
# def test_wallet_spend_cash_raise_except(twentywallet):
#     with pytest.raises(Insuffexcep):
#         twentywallet.spend_cash(100)