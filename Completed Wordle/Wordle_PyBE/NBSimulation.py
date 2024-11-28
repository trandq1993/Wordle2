import random
from collections import Counter
import pandas as pd
from collections import defaultdict
import csv
import WorldleBackEnd

# Read the word list from the CSV file
df = pd.read_csv('words.csv')
word_list = df.iloc[:, 0].str.strip().str.lower().values


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

def NBSimulate(solution, firstGuess):

    NBOutputCount = [0,0,0,0,0,0,0]
    valid_words = word_list.copy()
    allGuess = []
    allFeedback = []

    solution = solution.lower()
    top_word = firstGuess

    for i in range(len(NBOutputCount)):
        guess = top_word

        if(top_word == solution or i == len(NBOutputCount)-1):
            NBOutputCount[i] = NBOutputCount[i] + 1;
            break;

        feedback = get_feedback(guess, solution)

        allFeedback.append(feedback)
        allGuess.append(guess)

        valid_words = [word for word in valid_words if word.lower() not in [item.lower() for item in allGuess]]

        for feedback, guess_word in zip(allFeedback, allGuess):
            for j, feedback_type in enumerate(feedback):
                letter = guess_word[j]
                if feedback_type == "correct":
                    valid_words = [word for word in valid_words if word[j] == letter]
                if feedback_type == "present":
                    valid_words = [word for word in valid_words if letter in word and word[j] != letter]
                if feedback_type == "absent":
                    valid_words = [word for word in valid_words if letter not in word]

        probabilities = {word: 1 / len(valid_words) for word in valid_words}

        for word in list(probabilities):
            # Compare feedback for each word in the word list
            if get_feedback(guess, word) != feedback:
                probabilities[word] = 0

        probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        total = sum(probabilities.values())

        if total > 0:
            probabilities = {word: prob / total for word, prob in probabilities.items()}

        if probabilities:
            top_word = next(iter(probabilities))
        else:
            top_word = guess
            print("Error: probabilities is empty")

    return NBOutputCount
