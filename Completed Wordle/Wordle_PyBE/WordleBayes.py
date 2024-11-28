import random
from collections import defaultdict
import csv
import pandas as pd

from WorldleBackEnd import WordTracker

# Read the word list from the CSV file
df = pd.read_csv('words.csv')
word_list = df.iloc[:, 0].str.strip().str.lower().values

class bayesWordList:
    # Static (class-level) variable
    using_wordList = word_list.copy()

    @classmethod
    def init_word(cls):
        cls.using_wordList = word_list.copy()

    
    @classmethod
    def show_words(cls):
        # Return the current list of words used
        return cls.using_wordList


def get_feedback(guess, solution):
    # Normalize to lowercase for consistent comparison
    guess = guess.lower()
    solution = solution.lower()
    feedback = []
    for g, s in zip(guess, solution):
        if g == s:
            feedback.append("correct")
        elif g in solution:
            feedback.append("present")
        else:
            feedback.append("absent")
    
    return feedback

def update_probabilities(guess, feedback):
    WordTracker.append_feedback(feedback)
    filteredWordList = [word for word in bayesWordList.using_wordList if word.lower() not in [item.lower() for item in WordTracker.show_words()]]
    for feedback, guess_word in zip(WordTracker.show_feedback(), WordTracker.show_words()):
        for i, feedback_type in enumerate(feedback):
            letter = guess_word[i]
            if feedback_type == "correct":
                filteredWordList = [word for word in filteredWordList if word[i] == letter]
            elif feedback_type == "present":
                filteredWordList = [word for word in filteredWordList if letter in word and word[i] != letter]
            elif feedback_type == "absent":
                filteredWordList = [word for word in filteredWordList if letter not in word]
    probabilities = {word: 1 / len(filteredWordList) for word in filteredWordList}

    for word in list(probabilities):
        # Compare feedback for each word in the word list
        if get_feedback(guess, word) != feedback:
            probabilities[word] = 0
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {word: prob / total for word, prob in probabilities.items()}

    return probabilities


def get_TopProb(solution, guess):

    solution = solution.lower()
    guess = guess.lower()

    WordTracker.add_word(guess)

    feedback = get_feedback(guess, solution)
    probability = update_probabilities(guess, feedback).items()

    # Sort by probability and return the top 10 words
    sortedProb = dict(sorted(probability, key=lambda item: item[1], reverse=True)[:10])
    sortedProb = {key: value for key, value in sortedProb.items() if value != 0}
    sortedProb = {key: f"{round(value * 100, 2)}%" for key, value in sortedProb.items()}

    if solution == guess or len(WordTracker.wordUsed) == 0:
        WordTracker.clear_list()
        bayesWordList.init_word()

    return str(sortedProb).replace("'", '"')


def update_probabilities(guess, feedback):
    WordTracker.append_feedback(feedback)
    filteredWordList = [word for word in bayesWordList.using_wordList if word.lower() not in [item.lower() for item in WordTracker.show_words()]]
    for feedback, guess_word in zip(WordTracker.show_feedback(), WordTracker.show_words()):
        for i, feedback_type in enumerate(feedback):
            letter = guess_word[i]
            if feedback_type == "absent":
                # Remove all words containing the 'absent' letter
                filteredWordList = [word for word in filteredWordList if letter not in word]
    probabilities = {word: 1 / len(filteredWordList) for word in filteredWordList}

    for word in list(probabilities):
        # Compare feedback for each word in the word list
        if get_feedback(guess, word) != feedback:
            probabilities[word] = 0
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {word: prob / total for word, prob in probabilities.items()}

    return probabilities
