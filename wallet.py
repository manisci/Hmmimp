
class Insuffexcep(Exception):
        pass

class Wallet(object):

	def __init__(self,initial_amount = 0):
		self.balance = initial_amount
	def add_cash(self,amount):
		self.balance += amount
	def spend_cash(self,amount):
		if amount > self.balance:
			raise Insuffexcep("Not enough balance availbalbe to spend {}".format(self.balance))
		self.balance -= amount

    	
        