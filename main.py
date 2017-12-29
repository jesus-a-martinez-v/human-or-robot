from prepare_train_and_test_sets import prepare_train_data, prepare_test_data, prepare_train_and_test_data
from run_on_test_set import run_on_test_set
from summarize_data import summarize_data
from train_best import train_best
from train_candidates import train_candidates
import os

_menu_message = """
Project runner. You can run each stage separately or all at once. Here are your options:
    - [0]: All. No interruptions.
    - [1]: Summarize Data.
    - [2]: Prepare training and test sets.
    - [3]: Train candidate classifiers.
    - [4]: Train top 3 candidate classifiers.
    - [5]: Run best classifier on the test set.
"""

_continue_message = """
DONE. Continue? 
    - [0] No
    - [1] Yes
"""

_stages = {1: ('Summarize Data', summarize_data),
          2: ('Prepare training and test sets', prepare_train_and_test_data),
          3: ('Train candidate classifiers', train_candidates),
          4: ('Train top 3 candidate classifiers', train_best),
          5: ('Run best classifier on the test set', run_on_test_set)}


def _clean():
    print("Cleaning old run files.")
    files = ["./data/merged_test.csv",
             "./data/merged_train.csv",
             "./data/submission.csv",
             "./best.p",
             "./out.txt",
             "./best_candidates.p",
             "./train.p",
             "./test.p"]

    for f in files:
        try:
            os.remove(f)
            print("Successfully deleted file", f)
        except:
            print("Couldn't delete file", f, "because it doesn't exist.")
            continue


def run():
    correct_input = False

    while not correct_input:

        user_input = input(_menu_message)
        user_input = user_input.strip()

        if not user_input.isnumeric():
            print("No numeric option provided. Try again")
            continue

        option = int(user_input)
        no_interruptions = False

        if option > 5:
            print("Unknown option " + str(option) + " please try again.")
            continue

        correct_input = True

        if option <= 1:
            _clean()

            if option == 0:
                option = 1
                no_interruptions = True

        for stage in range(option, 6):
            stage_name, stage_function = _stages[stage]
            print("Running stage " + str(stage) + ":", stage_name)
            stage_function()

            if no_interruptions:
                continue

            continue_option = int(input(_continue_message))
            if not continue_option:
                break


if __name__ == "__main__":
    run()
