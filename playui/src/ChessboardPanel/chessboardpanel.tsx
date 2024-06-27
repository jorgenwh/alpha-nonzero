import './chessboardpanel.css';

import { Chessboard } from 'react-chessboard';
import { Arrow } from 'react-chessboard/dist/chessboard/types';
import { BoardOrientation } from 'react-chessboard/dist/chessboard/types';

interface ChessboardPanelProps {
    fen: string;
    orientation: BoardOrientation;
    onPieceDrop: (fromSquare: string, toSquare: string) => boolean;
    top5: string[][] | undefined;
    materialDifference: number;
}

const createCustomArrows = (top5: string[][] | undefined) => {
    if (top5 === undefined) return [];

    const maxValue = Math.max(...top5.map((k) => Number.parseFloat(k[1])));
    return top5.map((k) => {
        const move = k[0];
        const value = Number.parseFloat(k[1]);

        const goodness = 155 - Math.round((value / maxValue) * 155);
        const fromSquare = move.slice(0, 2);
        const toSquare = move.slice(2, 4);

        return [
            fromSquare, 
            toSquare, 
            `rgb(${goodness}, ${goodness}, ${goodness})`
        ] as Arrow;
    });
}

const buildChessboard = (
    fen: string, 
    orientation: BoardOrientation, 
    onPieceDrop: (fromSquare: string, toSquare: string) => boolean,
    top5: string[][] | undefined 
) => {
    const customArrows = createCustomArrows(top5);
    return (
        <Chessboard 
            position={fen} 
            boardOrientation={orientation} 
            onPieceDrop={onPieceDrop}
            customArrows={customArrows as Arrow[]}
        />
    );
}

const ChessboardPanel = ({ fen, orientation, onPieceDrop, top5, materialDifference }: ChessboardPanelProps) => {

    const chessboard = buildChessboard(fen, orientation, onPieceDrop, top5);
    const materialDifferenceDisplay = 'Material: ' + ((orientation === 'white') ? materialDifference : -materialDifference);

    return (
        <div className="ChessboardPanel">
            {chessboard}
            <div className="MaterialDifferenceLabel">{materialDifferenceDisplay}</div>
        </div>
    );
}

export default ChessboardPanel;
