import { useState, useEffect } from 'react';
import './App.css';

import { Chess } from 'chess.js';
import { BoardOrientation } from 'react-chessboard/dist/chessboard/types';

import SettingsPanel from './SettingsPanel/settingspanel';
import ChessboardPanel from './ChessboardPanel/chessboardpanel';
import AnatomyPanel from './AnatomyPanel/anatomypanel';


const kQueryBaseUrl = 'http://localhost:5000';
const kStartFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';


const queryPost = async (fen: string, rollouts: number) => {
    console.log("Posting query: " + fen + " with rollouts: " + rollouts);
    return fetch(kQueryBaseUrl + '/post', {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            fen: fen,
            rollouts: rollouts
        })
    })
    .then((response) => response.json())
    .then((response) => { return response.success })
    .catch((err) => console.log(err))
}

const queryGet = async () => {
    return fetch(kQueryBaseUrl + '/get')
    .then((response) => response.json())
    .then((response) => response)
    .catch((err) => console.log(err))
}


const App = () => {
    const [loading, setLoading] = useState<boolean>(false);
    const [game, setGame] = useState<Chess>(new Chess(kStartFen));
    const [fen, setFen] = useState<string>(kStartFen);
    const [orientation, setOrientation] = useState<BoardOrientation>('white');
    const [history, setHistory] = useState<string[]>([kStartFen]);
    const [rollouts, setRollouts] = useState<number>(100);
    const [value, setValue] = useState<number | undefined>(undefined);
    const [top5, setTop5] = useState<string[][] | undefined>(undefined);
    const [playAsWhite, setPlayAsWhite] = useState<boolean>(false);
    const [playAsBlack, setPlayAsBlack] = useState<boolean>(true);

    const clearEvaluation = () => {
        console.log("Clearing evaluation");
        setValue(undefined);
        setTop5(undefined);
    }

    const onPieceDrop = (fromSquare: string, toSquare: string) => {
        if (loading) return false;
        if (game.isGameOver()) return false;

        const gameCopy = new Chess(game.fen());
        try {
            gameCopy.move({
                from: fromSquare,
                to: toSquare,
                promotion: 'q'
            });
        }
        catch (e) {
            return false;
        }
        setFen(gameCopy.fen());
        setHistory([...history, gameCopy.fen()]);
        setGame(gameCopy);
        clearEvaluation();
        return true;
    }

    const onFenChange = (fen: string) => {
        if (loading) return;

        setFen(fen);
        const gameCopy = new Chess(game.fen());
        try {
            gameCopy.load(fen);
        }
        catch (e) {
            return;
        }
        setHistory([...history, gameCopy.fen()]);
        setGame(gameCopy);
        clearEvaluation();
    }

    const onFlip = () => {
        if (loading) return;

        if (orientation === 'white') {
            setOrientation('black');
        } 
        else {
            setOrientation('white');
        }
    }

    const onMovePop = () => {
        if (loading) return;

        if (history.length > 1) {
            const historyCopy = [...history];
            historyCopy.pop();
            const newFen = historyCopy[historyCopy.length - 1];
            setHistory(historyCopy);
            setFen(newFen);
            setGame(new Chess(newFen));
            clearEvaluation();
        }
    }

    const onResetGame = () => {
        if (loading) return;

        setHistory([kStartFen]);
        setFen(kStartFen);
        setGame(new Chess(kStartFen));
        clearEvaluation();
    }

    const gameOutcome = () => {
        if (game.isDraw()) return 'draw';
        if (game.isCheckmate()) return 'checkmate';
        if (game.isStalemate()) return 'stalemate';
        if (game.isInsufficientMaterial()) return 'insufficient material';
        if (game.isGameOver()) return 'game over';
        return undefined;
    }

    const filterTop5 = (newTop5: string[][]) => {
        const filteredTop5 = newTop5.filter((k) => {
            const value = Number.parseFloat(k[1]);
            return (value !== 0);
        });

        return filteredTop5;
    }

    const evaluate = async () => {
        if (loading) return;
        if (game.isGameOver()) return;

        setLoading(true);

        const post = await queryPost(game.fen(), rollouts);
        if (!post) {
            console.log('Failed to post query:' + post);
            setLoading(false);
            return false;
        }

        const get = await queryGet();
        const newValue = get.value;
        const newTop5 = filterTop5(get.top5);

        console.log("Setting value, top5");
        setValue(newValue);
        setTop5(newTop5);
        setLoading(false);

        return true;
    }

    const getMaterialDifference = () => {
        const board = game.board();
        let materialDifference = 0;

        for (let i = 0; i < board.length; i++) {
            for (let j = 0; j < board[i].length; j++) {
                const piece = board[i][j];
                if (piece !== null) {
                    const type = piece.type;
                    const color = piece.color;

                    if (type === 'p') {
                        materialDifference += (color === 'w') ? 1 : -1;
                    }
                    if (type === 'n') {
                        materialDifference += (color === 'w') ? 3 : -3;
                    }
                    if (type === 'b') {
                        materialDifference += (color === 'w') ? 3 : -3;
                    }
                    if (type === 'r') {
                        materialDifference += (color === 'w') ? 5 : -5;
                    }
                    if (type === 'q') {
                        materialDifference += (color === 'w') ? 9 : -9;
                    }
                }
            }
        }

        return materialDifference;
    }

    const playAiMove = (index: number) => {
        if (loading) { 
            console.log("Failed to play AI move: loading");
            return;
        }
        if (top5 === undefined) {
            console.log("Failed to play AI move: top5 undefined");
            return;
        }
        if (index >= top5.length) {
            console.log("Failed to play AI move: index out of bounds for top5");
            return;
        }
        if (game.isGameOver()) {
            console.log("Failed to play AI move: game is over");
            return;
        }

        const move = top5[index][0];
        const fromSquare = move.slice(0, 2);
        const toSquare = move.slice(2, 4);
        const promotion = (move.length == 5) ? move.slice(4, 5) : undefined;

        console.log("Playing AI move: " + move);
        const gameCopy = new Chess(game.fen());
        try {
            gameCopy.move({
                from: fromSquare,
                to: toSquare,
                promotion: promotion
            });
        }
        catch (e) { }
        setFen(gameCopy.fen());
        setHistory([...history, gameCopy.fen()]);
        setGame(gameCopy);
        setTop5(undefined);
    }

    useEffect(() => {
        if (game.isGameOver()) return;

        if (playAsWhite && game.turn() === 'w') {
            if (top5 === undefined) {
                evaluate();
            }
            else {
                playAiMove(0);
            }
        }
        if (playAsBlack && game.turn() === 'b') {
            if (top5 === undefined) {
                evaluate();
            }
            else {
                playAiMove(0);
            }
        }
    }, [game, value, top5, playAsWhite, playAsBlack]);

    const displayedValue = (value === undefined) ? 'N/A' : value.toFixed(3);
    const materialDifference = getMaterialDifference();
    let status = gameOutcome();
    if (status === undefined) {
        status = (loading) ? 'loading...' : 'ready';
    }

    return (
        <div className="App">
            <div className="SettingsPanelContainer">
                <SettingsPanel
                    loading={loading}
                    fen={fen} 
                    onFenChange={onFenChange}
                    onFlip={onFlip}
                    onMovePop={onMovePop}
                    onResetGame={onResetGame}
                    rollouts={rollouts}
                    setRollouts={setRollouts}
                    playAsWhite={playAsWhite}
                    setPlayAsWhite={setPlayAsWhite}
                    playAsBlack={playAsBlack}
                    setPlayAsBlack={setPlayAsBlack}
                />
            </div>
            <div className="ChessboardPanelAnatomyPanelContainer">
                <ChessboardPanel 
                    fen={game.fen()} 
                    orientation={orientation as BoardOrientation}
                    onPieceDrop={onPieceDrop}
                    top5={top5}
                    materialDifference={materialDifference}
                />
                <AnatomyPanel
                    evaluate={evaluate}
                    value={displayedValue}
                    top5={top5}
                />
            </div>
            <div className="Footer">Status: {status}</div>
        </div>
    );
}

export default App;
