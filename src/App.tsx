import React, { useState, useEffect, useRef } from "react";
import Papa from "papaparse";
import WordGrid from "./components/WordGrid";
import './App.css';

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

    useEffect(() => {
        fetchWordsFromCSV();
        fetchAllWordsFromCSV();

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

        window.addEventListener('keydown', handleKeyDown);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [gameOver]);

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

    return (
        <div className="App">
            <h1>ML Wordle 2.0</h1>
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
    );
};

export default App;
