import React from "react";

interface WordGridProps {
    guesses: string[];
    solution: string;
    currentGuess: string;
}

const WordGrid: React.FC<WordGridProps> = ({ guesses, solution, currentGuess }) => {
    const getLetterStatus = (letter: string, index: number) => {
        if (solution[index] === letter) return "correct";
        if (solution.includes(letter))
        {
            return "present"
        };
        return "absent";
    };

    const totalRows = 6; // Total number of rows (guesses)
    const currentGuessRow = guesses.length; // The index of the row for the current guess

    return (
        <div className="word-grid">
            {[...Array(totalRows)].map((_, guessIndex) => (
                <div className="word-row" key={guessIndex}>
                    {guessIndex < guesses.length
                        ? guesses[guessIndex].split("").map((letter, letterIndex) => (
                            <span
                                key={letterIndex}
                                className={`word-letter ${getLetterStatus(letter, letterIndex)}`}
                            >
                                {letter}
                            </span>
                        ))
                        : guessIndex === currentGuessRow
                            ? currentGuess.split("").map((letter, letterIndex) => (
                                <span
                                    key={letterIndex}
                                    className="word-letter-current"
                                >
                                    {letter}
                                </span>
                            ))
                            : Array(5).fill("").map((_, letterIndex) => (
                                <span key={letterIndex} className="word-letter">
                                    {/* Render empty span for unused letters */}
                                </span>
                            ))}
                    {/* Always render 5 letters, including empty spans for the current guess row */}
                    {guessIndex === currentGuessRow && currentGuess.length < 5 && (
                        Array.from({ length: 5 - currentGuess.length }, (_, letterIndex) => (
                            <span key={letterIndex + currentGuess.length} className="word-letter" />
                        ))
                    )}
                </div>
            ))}
        </div>
    );
};

export default WordGrid;
