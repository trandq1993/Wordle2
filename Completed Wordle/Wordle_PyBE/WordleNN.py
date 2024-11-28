import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import json
import WorldleBackEnd

class WordleNNSV:
    # Static variable (class attribute)
    guesses = []

    def __init__(self, solution):
        self.solution = solution

    @classmethod
    def add_guess(self, guess):
        self.guesses.append(guess)

    @classmethod
    def initGuess(self):
        self.guesses = []

# Load the word list
with open("words.csv") as f:
    word_list = [word.strip().lower() for word in f.readlines() if len(word.strip()) == 5]

# Define the custom vectorize function
def custom_vectorize(word_list):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    vectorized = []
    for word in word_list:
        word_vector = []
        for letter in word:
            letter_vector = [0] * 26
            letter_vector[alphabet.index(letter)] = 1
            word_vector.extend(letter_vector)  # Append the one-hot vector for each letter
        vectorized.append(word_vector)  # Resulting vector will be of size 130
    return np.array(vectorized)

# Define the model
class WordleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(WordleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def load_model():
    """Load the pre-trained model."""
    model = WordleNN(input_size=130, output_size=15)  # Adjust input_size/output_size as needed
    try:
        model.load_state_dict(torch.load("wordle_nn.pth"))
        model.eval()
        return model
    except FileNotFoundError:
        print("Model file not found. Please ensure the model is trained and saved as 'wordle_nn.pth'.")
        exit()

# Filter valid words based on feedback
def filter_valid_words(valid_words, guesses):
    excluded_letters = set()
    for guess, feedback in guesses:
        for i, (color, letter) in enumerate(feedback):
            if color == "green":
                valid_words = [word for word in valid_words if word[i] == letter]
            elif color == "yellow":
                valid_words = [word for word in valid_words if letter in word and word[i] != letter]
            elif color == "gray":
                if not any(fb[0] in {"green", "yellow"} and fb[1] == letter for fb in feedback):
                    excluded_letters.add(letter)
    valid_words = [word for word in valid_words if all(letter not in word for letter in excluded_letters)]
    return valid_words

# Provide feedback for the guess
def give_feedback(guess, target_word):
    target_word = target_word.lower()
    feedback = []  # (color, letter)
    target_word_list = list(target_word)

    for i, char in enumerate(guess):
        if char == target_word[i]:
            feedback.append(('green', char))
            target_word_list[i] = None
        elif char in target_word_list:
            feedback.append(('yellow', char))
            target_word_list[target_word_list.index(char)] = None
        else:
            feedback.append(('gray', char))
    return feedback

# Suggest the top words
def suggest_top_words(model, valid_words, guesses, top_n=3):

    valid_words = filter_valid_words(valid_words, guesses)
    if not valid_words:
        return [("NO SUGGESTIONS", 0.0)] * top_n
    vectorized = custom_vectorize(valid_words)
    vectorized_tensor = torch.tensor(vectorized, dtype=torch.float32)
    outputs = model(vectorized_tensor)
    outputs = outputs.view(len(valid_words), 5, 3)
    probabilities = torch.softmax(outputs, dim=2)[:, :, 2].mean(dim=1)
    top_indices = torch.argsort(probabilities, descending=True)[:min(top_n, len(valid_words))]
    suggestions = [(valid_words[i], probabilities[i].item()) for i in top_indices]
    while len(suggestions) < top_n:
        suggestions.append(("NO SUGGESTIONS", 0.0))
    return suggestions

def get_NNOutput(guess, solution):
    guess = guess.lower()
    solution = solution.lower()
    model = load_model()

    valid_words = word_list.copy()

    feedback = give_feedback(guess, solution)
    WordleNNSV.guesses.append((guess, feedback))

    valid_words = [word for word in valid_words if word != guess]
    top_suggestions = suggest_top_words(model, valid_words,  WordleNNSV.guesses, top_n=3)

    suggestions_json = [
        {suggestion.lower(): f"{prob * 100:.2f}%"}
        for idx, (suggestion, prob) in enumerate(top_suggestions, start=1)
    ]

    merged_dict = {k: v for d in suggestions_json for k, v in d.items()}
    merged_dict = {key: value for key, value in merged_dict.items() if value != "0.00%"}
    single_json_value = json.dumps(merged_dict)
    
    if (guess == solution or len(WordleNNSV.guesses) == 6) and guess != "":
        WordleNNSV.guesses = []

    return( str(single_json_value))

def NNSimulate(solution, firstGuess):

    model = load_model()
    NNOutputCount = [0,0,0,0,0,0,0]
    valid_words = word_list.copy()
    allGuesses = []

    top_word = firstGuess.lower();
    solution = solution.lower()

    for i in range(len(NNOutputCount)):
        guess = top_word
   
        if(guess == solution or i == len(NNOutputCount)-1):
            NNOutputCount[i] = NNOutputCount[i] + 1;
            break;
        if (guess == "no suggestions"):
            break;
        feedback = give_feedback(guess, solution)
        allGuesses.append((guess, feedback))
        valid_words = [word for word in valid_words if word != guess]
        top_suggestions = suggest_top_words(model, valid_words,  allGuesses, top_n=3)
        top_word = top_suggestions[0][0].lower()
    return NNOutputCount




