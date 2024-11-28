from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import threading

import pandas as pd

import json
import asyncio

import random
import WordleBayes
import WordleNN
import NBSimulation


df = pd.read_csv('words.csv')
word_list = list(df.iloc[:, 0].str.strip().str.lower().values)

app = FastAPI()


# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WordTracker:
    # Static variable (class-level) initialized as an empty list
    wordUsed = []
    feedbackUsed = []
    NNSimArray = [0,0,0,0,0,0,0]
    NBSimArray = [0,0,0,0,0,0,0]


    @classmethod
    def setNBSimArray(cls,array):
        cls.NNSimArray = array

    @classmethod
    def show_NBSimArray(cls):
        return cls.NNSimArray

    @classmethod
    def add_word(cls, word):
        # If the word is already in the list, do not add it
        if word in cls.wordUsed:
            return  # Early return if the word is already present

        # If the list is full (6 words), clear the list before adding the new word
        if len(cls.wordUsed) == 6:
            cls.wordUsed.clear()

        # Append the new word
        cls.wordUsed.append(word)

    @classmethod
    def show_words(cls):
        return cls.wordUsed

    @classmethod
    def append_feedback(cls, feedback):

        # If the list is full (6 words), clear the list before adding the new word
        if len(cls.feedbackUsed) == 6:
            cls.feedbackUsed.clear()

        cls.feedbackUsed.append(feedback)

    @classmethod
    def show_feedback(cls):
        return cls.feedbackUsed

    @classmethod
    def clear_list(cls):
        cls.wordUsed.clear()
        cls.feedbackUsed.clear()

# Shared global variable
NBArray = [0,0,0,0,0,0,0]
NNArray = [0,0,0,0,0,0,0]

class Message(BaseModel):
    content: str

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    #NBSimulation.NBSimulate(random.choices(word_list, k=1000))
    while True:
        data = { str(NNArray)+"!"+str(NBArray)}
        try:
            await websocket.send_text(data)
        except Exception as e:
            pass  
        # Wait for a few seconds before sending the next update
        await asyncio.sleep(1)  # Update every 3 seconds

@app.post("/simulate")
async def handle_ctrl_s(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_simulation)


def run_simulation():
    global NNArray
    global NBArray

    NNOutputCount = [0,0,0,0,0,0,0]
    NBOutputCount = [0,0,0,0,0,0,0]

    randomSolutions = random.choices(word_list, k=10000)

    for solution in randomSolutions:
        firstGuess = random.choice(word_list).lower()
        NBSimulationArray = NBSimulation.NBSimulate(solution, firstGuess)
        NNSimulationArray = WordleNN.NNSimulate(solution, firstGuess)

        if (firstGuess == solution.lower()):
            NNOutputCount[0] = NNOutputCount[0] + 1;
            NBOutputCount[0] = NBOutputCount[0] + 1;
        else:
            NBOutputCount = [a + b for a, b in zip(NBOutputCount, NBSimulationArray)]
            NNOutputCount = [a + b for a, b in zip(NNOutputCount, NNSimulationArray)]

        NBArray = NBOutputCount
        NNArray = NNOutputCount

@app.post("/send_message")
async def send_message(message: Message):

    solution = message.content.split(',')[0]
    guess = message.content.split(',')[1]
    return {"received_message": (WordleBayes.get_TopProb(solution, guess) 
                                 + "!" + WordleNN.get_NNOutput(guess, solution)).replace("'", '"')}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
   
