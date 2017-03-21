from __future__ import print_function
import time
import sys
import os

home_path = os.path.expanduser("~")

def join(*args):
	if len(args) == 0:
		args = ['']
	str_args = []
	for a in args:
		str_args.append(str(a))
	return os.path.join(*str_args)

def homeDir(*args):
	return join(home_path,join(*args))

def desktopDir(*args):
    return homeDir('Desktop',join(*args))

class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.__update_amount(0)

    def animate(self, iter):
        print('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

class Timer:
    def __init__(self, time_s):
        self.time_s = time_s
        self.start_time = time.time()
    def check(self):
        if time.time() - self.start_time > self.time_s:
            return True
        else:
            return False
    def reset(self):
        self.start_time = time.time()
