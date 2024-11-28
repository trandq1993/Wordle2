import React, { useState, useEffect, useRef } from "react";

import Papa from "papaparse";

import WordGrid from "./components/WordGrid";
import NBHistogram from './components/NBHistogram'; 
import NNHistogram from './components/NNHistogram'; 

import { Bar } from 'react-chartjs-2';

import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

import './App.css';

import axios from "axios";


const App: React.FC = () => {
    const [guesses, setGuesses] = useState<string[]>([]);
    const [currentGuess, setCurrentGuess] = useState<string>("");
    const [gameOver, setGameOver] = useState<boolean>(false);
    const [solutionWord, setSolutionWord] = useState<string>("");
    const [message, setMessage] = useState<string>("");
    const [validWords, setValidWords] = useState<string[]>([]);

    const submitRef = useRef<HTMLButtonElement>(null);

    const getRandomWord = (wordsArray: string[]) => {
        const randomIndex = Math.floor(Math.random() * wordsArray.length);
        return wordsArray[randomIndex];
    };

    const [socket, setSocket] = useState(null);
    const [connectionStatus, setConnectionStatus] = useState("Connecting...");

    const [responseMessage, setResponseMessage] = useState("");
    const [bayesDictionary, setBayesDictionary] = useState<Record<string, string>>({});
    const [nnDictionary, setNNDictionary] = useState<Record<string, string>>({});
    const [NNSimulate, setNNSimulate] = useState<any>(null);
    const [NBSimulate, setNBSimulate] = useState<any>(null);

    const [NNScore, setNNScore] = useState("");
    const [NBScore, setNBScore] = useState("");

    const [NNAccuracy, setNNAccuracy] = useState("");
    const [NBAccuracy, setNBAccuracy] = useState("");

    ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);


    useEffect(() => {
        fetchWordsFromCSV();
        fetchAllWordsFromCSV();

        const socket = new WebSocket('ws://localhost:8000/ws');
        const connectWebSocket = () => {
            const socket = new WebSocket("ws://localhost:8000/ws");

            socket.onopen = () => {
                setConnectionStatus("Connected");
                console.log("WebSocket connected");
            };
            
            socket.onmessage = (event) => {
                handleSocketMessage(event);
            };

            socket.onerror = (error) => {
                console.error("WebSocket Error:", error);
                setConnectionStatus("Error connecting...");
            };

            socket.onclose = () => {
                setConnectionStatus("Connection closed. Retrying...");
                // Retry after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };

            setSocket(socket);
        };

        connectWebSocket();

        const handleKeyPress = (event) => {
            if (event.ctrlKey && event.key === 's') {
                event.preventDefault(); // Prevents the default browser save behavior
                // Call your backend API here
                fetchData();
            }
        };

        const handleKeyDown = (event: KeyboardEvent) => {
            if (!gameOver) {
                if (event.key === 'Enter') {
                    submitRef.current?.click()

                } else if (event.key.length === 1 && event.key.match(/[a-zA-Z]/)) {
                    setCurrentGuess((prev) => (prev.length < 5 ? prev + event.key.toUpperCase() : prev));
                } else if (event.key === 'Backspace') {
                    setCurrentGuess((prev) => prev.slice(0, -1));
                }
            }
        };


        window.addEventListener('keydown', handleKeyPress);
        window.addEventListener('keydown', handleKeyDown);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keydown', handleKeyPress);
            socket.close();
        };
    }, [gameOver]);

    const handleSocketMessage = (event: MessageEvent) => {

        const [nnResponse, nbResponse] = event.data.split("!");

        const nbArrayResp = JSON.parse(nbResponse)
        const nnArrayResp = JSON.parse(nnResponse)

        const nbChartData = generateChartData(nbArrayResp)
        const nnChartData = generateChartData(nnArrayResp)


        setNBSimulate(nbChartData)
        setNNSimulate(nnChartData)
        setNBScore(weightedAverage(nbArrayResp))
        setNNScore(weightedAverage(nnArrayResp))
        setNBAccuracy(accuracyCalc(nbArrayResp))
        setNNAccuracy(accuracyCalc(nnArrayResp))
    };

    const accuracyCalc = (array: number[]) => {
        const failed = array[6];
        const total = array.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
        const accuracyPercent = String(parseFloat((100-(failed / total)).toFixed(5)) + "%");
        return accuracyPercent;
    };

    const weightedAverage = (array : number[]) => {
        // Define the weights
        const weights = [0, 6, 5, 4, 3, 2, 0];

        // Calculate the weighted sum
        const weightedSum = array.reduce((sum, value, index) => {
            return sum + value * weights[index];
        }, 0);

        // Calculate the total weight (sum of weights)
        const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);

        // Return the weighted average
        return String((weightedSum / totalWeight).toFixed(2));
    };
    const fetchData = () => {
        fetch('http://localhost:8000/simulate', { method: 'POST' }).catch(error => {
            console.error('Error calling backend:', error);
        });
    };

    const sendMessage = async (message, callback) => {
        try {
            // Send the message to FastAPI
            const response = await axios.post('http://localhost:8000/send_message', { content: message });

            // Update the response message state with the server's response
            const receivedMessage = response.data.received_message;
            const [bayesResponse, nnResponse] = receivedMessage.split("!");

            //StaticVariable.bayesJSON = receivedMessage;
            setResponseMessage(receivedMessage);
            setBayesDictionary(JSON.parse(bayesResponse));
            setNNDictionary(JSON.parse(nnResponse))

            // Call the provided callback with the received message
            if (callback && typeof callback === 'function') {
                callback(receivedMessage);
            }

        } catch (error) {
            console.error('Error sending message:', error);
        }
    };

    const handleServerResponse = (message) => {
    };


    const fetchWordsFromCSV = async () => {
        const response = await fetch('/words.csv');
        const reader = response.body?.getReader();
        const result = await reader?.read();
        const decoder = new TextDecoder('utf-8');
        const csvText = decoder.decode(result?.value);
        Papa.parse(csvText, {
            complete: (results) => {
                const wordsArray = results.data.flat().map((word: string) => word.toUpperCase());
                const randomWord = getRandomWord(wordsArray); 
                setSolutionWord(randomWord.toUpperCase());
            },
            header: false
        });
    };

    const fetchAllWordsFromCSV = async () => {
        const response = await fetch('/allwords.csv');
        const reader = response.body?.getReader();
        const result = await reader?.read();
        const decoder = new TextDecoder('utf-8');
        const csvText = decoder.decode(result?.value);

        Papa.parse(csvText, {
            complete: (results) => {
                const wordsArray = results.data.flat().map((word: string) => word.toUpperCase());
                setValidWords(wordsArray);
            },
            header: false

        });
    };

    const handleSubmit = () => {
        if (currentGuess.length !== 5) {
            setMessage("Please enter a 5-letter word.");
            return;
        }

        if (!validWords.includes(currentGuess)) {
            setMessage("Invalid word! Please enter a valid 5-letter word.");
            return;
        }

        sendMessage((solutionWord + "," + currentGuess), handleServerResponse);
        const updatedGuesses = [...guesses, currentGuess];
        setGuesses(updatedGuesses);
        setCurrentGuess("");

        if (currentGuess === solutionWord) {
            setGameOver(true);
            setMessage("Congratulations! You've guessed the word.");
            setTimeout(resetGame, 1000);
        } else if (updatedGuesses.length >= 6) {
            setGameOver(true);
            setMessage(`Game over! The word was ${solutionWord}.`);
            setTimeout(resetGame, 1000);
        }
    };

    const resetGame = () => {
        setGuesses([]);
        setCurrentGuess("");
        setGameOver(false);
        setMessage("");
        fetchWordsFromCSV();
    };

    // Chart options with title and axis labels
    const chartOptions = {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: 'Completion Counts',
            },
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Completion Count', // x-axis becomes the 'Completion Count'
                },
                beginAtZero: true,
                ticks: {
                    stepSize: 5, // Adjust step size to suit your data scale
                    callback: function (value: number) {
                        // Convert negative values to positive by using Math.abs
                        return Math.abs(value);
                    },
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Attempts', // y-axis becomes the 'Attempts'
                },
            },
        },
        indexAxis: 'y', // This makes the chart horizontal
    };

    const generateChartData = (arrayData: number[]) => {
        const labels = ["1", "2", "3", "4", "5", "6", "Failed"];
        const data = Object.values(arrayData);

        return {
            labels,
            datasets: [
                {
                    label: 'Completion Count',
                    data,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                },
            ],
        };
    };

    return (

        <div style={{ display: "flex", alignItems: "center" }}>
            <ul>
                <h3>Nueral Network Simulation</h3>
                <h5>Performance Score: {NNScore}</h5>
                <h5>Accuracy Score: {NNAccuracy}</h5>
                <div style={{ width: '500px', height: '300px', margin: 'auto' }}>
                    {NNSimulate ? <Bar data={NNSimulate} options={chartOptions} /> : <p>Loading chart...</p>}
                </div>
                <h3>Naive Bayes Simulation</h3>
                <h5>Performance Score: {NBScore}</h5>
                <h5>Accuracy Score: {NBAccuracy}</h5>
                <div style={{ width: '500px', height: '300px', margin: 'auto' }}>
                    {NBSimulate ? <Bar data={NBSimulate} options={chartOptions} /> : <p>Loading chart...</p>}
                </div>
            </ul>
            
            <div style={{borderLeft: "2px solid #ccc", height: "500px", margin: "0 10px"}}></div>
            <div className="App">
                <h1>ML Wordle 2</h1>
                {solutionWord && (
                    <WordGrid
                        guesses={guesses}
                        solution={solutionWord}
                        currentGuess={currentGuess}
                    />
                )}
                <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
                    <button ref={submitRef} type="submit" disabled={currentGuess.length !== 5 || gameOver}>
                        Submit
                    </button>
                </form>

                {message && <div className="message">{message}</div>}
            </div>
            <div style={{ borderLeft: "2px solid #ccc", height: "500px", margin: "0 10px" }}></div>

            <div>
                <h2>Naive Bayes</h2>
                <ul>
                    {Object.entries(bayesDictionary).map(([key, value]) => (
                        <p key={key}>
                            {key}: {value}
                        </p>
                    ))}
                </ul>
            </div>
            <div style={{ borderLeft: "3px solid #DDD", height: "0px", margin: "0 10px" }}></div>
            <div>
                <h2>Neural Network</h2>
                <ul>
                    {Object.entries(nnDictionary).map(([key, value]) => (
                        <p key={key}>
                            {key}: {value}
                        </p>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default App;
