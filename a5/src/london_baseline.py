# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

# the list comprehension in the below line is to accommodate lines with just the newline character
total, correct = utils.evaluate_places('birth_dev.tsv', ['London'] * len([line for line in open('birth_dev.tsv').read().split('\n') if line]))

if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))