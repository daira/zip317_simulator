#!/usr/bin/env python3

from secrets import randbelow
from collections import deque
from math import floor
from pprint import PrettyPrinter

MARGINAL_FEE = 5000
GRACE_ACTIONS = 2
WEIGHT_CAP = 4.0
MAX_STANDARD_TX_SIGOPS = 4000
MAX_BLOCK_SIGOPS = 20000
MAX_BLOCK_SIZE = 2000000
MEMPOOL_COST_LIMIT = 80000000
BLOCK_UNPAID_ACTION_LIMIT = 50

FIXED_ONE = 1000000000000

class Block:
    def __init__(self):
        self.txns = deque()
        self.size = 0
        self.sigops = 0

    def add_if_fits(self, tx):
        if self.size + tx.size <= MAX_BLOCK_SIZE and self.sigops + tx.sigops <= MAX_BLOCK_SIGOPS:
            self.txns.append(tx)
            self.size += tx.size
            self.sigops += tx.sigops
            return True
        else:
            return False

    def health(self, extra_tx=None):
        num = sum((min(tx.fee, tx.conventional_fee) for tx in self.txns))
        den = sum((tx.conventional_fee for tx in self.txns))
        if extra_tx is not None:
            num += min(extra_tx.fee, extra_tx.conventional_fee)
            den += extra_tx.conventional_fee

        return num/den

    def total_unpaid_actions(self):
        return sum((tx.unpaid_actions for tx in self.txns))


class Mempool:
    def __init__(self, txns):
        self.txns = deque()
        w = 0
        for tx in txns:
            w += tx.scaled_weight
            self.txns.append((w, tx))
        self.total_scaled_weight = w
        assert len(self.txns) == 0 or self.txns[-1][0] == self.total_scaled_weight

    def pick_by_weight(self):
        assert len(self.txns) > 0
        assert self.txns[-1][0] == self.total_scaled_weight
        target = randbelow(self.total_scaled_weight)
        new_txns = deque()
        found = None
        new_w = 0
        for (w, tx) in self.txns:
            if found is None and w > target:
                found = tx
            else:
                new_w += tx.scaled_weight
                new_txns.append((new_w, tx))

        assert found is not None
        self.txns = new_txns
        self.total_scaled_weight = new_w
        assert len(self.txns) == 0 or self.txns[-1][0] == self.total_scaled_weight
        return found

    def num_transactions(self):
        return len(self.txns)

    def total_fees(self):
        return sum((tx.fee for (_, tx) in self.txns))

    def total_conventional_fees(self):
        return sum((tx.conventional_fee for (_, tx) in self.txns))

    def total_scaled_weight(self):
        return sum((tx.scaled_weight for (_, tx) in self.txns))


def conventional_fee(logical_actions):
    return MARGINAL_FEE * max(GRACE_ACTIONS, logical_actions)

class Tx:
    def __init__(self, size, sigops, fee, logical_actions):
        assert isinstance(size, int) and 0 <= size and size <= MAX_BLOCK_SIZE
        assert isinstance(sigops, int) and 0 <= sigops and sigops <= MAX_STANDARD_TX_SIGOPS
        assert isinstance(fee, int) and 0 <= fee
        assert isinstance(logical_actions, int) and 1 <= logical_actions
        self.size = size
        self.sigops = sigops
        self.fee = fee
        self.logical_actions = logical_actions
        self.conventional_fee = conventional_fee(logical_actions)
        self.scaled_weight = min((FIXED_ONE * max(1, self.fee)) // self.conventional_fee, int(FIXED_ONE * WEIGHT_CAP))
        self.cost = max(size, 4000)
        self.unpaid_actions = max(0, max(GRACE_ACTIONS, logical_actions) - fee//MARGINAL_FEE)

    def is_low_fee(self):
        return self.scaled_weight < FIXED_ONE

    def __repr__(self):
        return ("Tx(size=%r, sigops=%r, fee=%r, logical_actions=%r, conventional_fee=%r, weight=%r, cost=%r, unpaid_actions=%r)"
                % (self.size, self.sigops, self.fee, self.logical_actions, self.conventional_fee, self.scaled_weight/FIXED_ONE, self.cost, self.unpaid_actions))

    @classmethod
    def random(cls):
        while True:
            if prob(FIXED_ONE//2):
                # similar to sandblasting txns
                logical_actions = 100 + randbelow(800)
                fee = (conventional_fee(logical_actions) * randbelow(FIXED_ONE//10))//FIXED_ONE
            elif prob(FIXED_ONE//10):
                # unconventional fee
                logical_actions = 1 + randbelow(50)
                fee = (conventional_fee(logical_actions) * randbelow(FIXED_ONE*2))//FIXED_ONE
            elif prob(FIXED_ONE//10):
                # ZIP 313 fee
                logical_actions = 1 + randbelow(20)
                fee = 1000
            else:
                # ZIP 317 fee
                logical_actions = 1 + randbelow(20)
                fee = conventional_fee(logical_actions)

            size = logical_actions*800 + randbelow(logical_actions*1000)
            sigops = randbelow(logical_actions + 10)
            try:
                return cls(size, sigops, fee, logical_actions)
            except AssertionError:
                pass


def shuffled(x):
    x = list(x)
    while len(x) > 0:
        i = randbelow(len(x))
        yield x[i]
        x[i] = x[-1]
        x.pop()

def prob(p):
    return randbelow(FIXED_ONE) <= p


def algorithm_1(block, mempool):
    high_fee_mempool = Mempool((tx for tx in mempool if not tx.is_low_fee()))
    while high_fee_mempool.num_transactions() > 0:
        block.add_if_fits(high_fee_mempool.pick_by_weight())

    remaining_mempool = Mempool((tx for tx in mempool if tx.is_low_fee()))
    remaining_block_size = MAX_BLOCK_SIZE - block.size
    print("block_size =", block.size)
    print("remaining_block_size =", remaining_block_size)
    size_target = block.size + (remaining_block_size * remaining_mempool.total_scaled_weight) // (FIXED_ONE * remaining_mempool.num_transactions())
    print("size_target =", size_target)

    while remaining_mempool.num_transactions() > 0:
        tx = remaining_mempool.pick_by_weight()
        if block.size + tx.size > size_target:
            break
        block.add_if_fits(tx)


def algorithm_2(block, mempool):
    high_fee_mempool = Mempool((tx for tx in mempool if not tx.is_low_fee()))
    while high_fee_mempool.num_transactions() > 0:
        block.add_if_fits(high_fee_mempool.pick_by_weight())

    remaining_mempool = Mempool((tx for tx in mempool if tx.is_low_fee()))
    remaining_block_size = MAX_BLOCK_SIZE - block.size
    print("block_size =", block.size)
    print("remaining_block_size =", remaining_block_size)
    size_target = block.size + (remaining_block_size * remaining_mempool.total_fees()) // remaining_mempool.total_conventional_fees()
    print("size_target =", size_target)

    while remaining_mempool.num_transactions() > 0:
        tx = remaining_mempool.pick_by_weight()
        if block.size + tx.size > size_target:
            break
        block.add_if_fits(tx)


def algorithm_3(block, mempool):
    high_fee_mempool = Mempool((tx for tx in mempool if not tx.is_low_fee()))
    while high_fee_mempool.num_transactions() > 0:
        block.add_if_fits(high_fee_mempool.pick_by_weight())

    for tx in shuffled((tx for tx in mempool if tx.is_low_fee())):
        if prob(tx.scaled_weight):
            if block.health(tx) < 2/3:
                break
            block.add_if_fits(tx)


def algorithm_4(block, mempool):
    """This is the proposed algorithm."""
    high_fee_mempool = Mempool((tx for tx in mempool if not tx.is_low_fee()))
    while high_fee_mempool.num_transactions() > 0:
        block.add_if_fits(high_fee_mempool.pick_by_weight())

    remaining_mempool = Mempool((tx for tx in mempool if tx.is_low_fee()))
    unpaid_actions = 0
    while remaining_mempool.num_transactions() > 0:
        tx = remaining_mempool.pick_by_weight()
        assert tx.unpaid_actions > 0
        if unpaid_actions + tx.unpaid_actions > BLOCK_UNPAID_ACTION_LIMIT:
            continue
        if block.add_if_fits(tx):
            unpaid_actions += tx.unpaid_actions



def test(alg):
    block = Block()
    n = randbelow(1000)
    mempool = deque()
    cost = 0
    while True:
        tx = Tx.random()
        if cost + tx.cost > MEMPOOL_COST_LIMIT:
            break
        cost += tx.cost
        mempool.append(tx)

    pp = PrettyPrinter(indent=2)
    print("Algorithm:", alg.__name__)
    print("Mempool:")
    pp.pprint(mempool)
    alg(block, mempool)
    print("Block (size=%r, sigops=%r):" % (block.size, block.sigops))
    pp.pprint(block.txns)
    print("Health:", block.health())
    print("Unpaid actions:", block.total_unpaid_actions())
    print("")

if __name__ == "__main__":
    #test(algorithm_1)
    #test(algorithm_2)
    #test(algorithm_3)
    test(algorithm_4)