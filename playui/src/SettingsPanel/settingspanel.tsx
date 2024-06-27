import './settingspanel.css';

interface SettingsPanelProps {
    loading: boolean;
    fen: string;
    onFenChange: (fen: string) => void;
    onFlip: () => void;
    onMovePop: () => void;
    onResetGame: () => void;
    rollouts: number;
    setRollouts: (rollouts: number) => void;
    playAsWhite: boolean;
    setPlayAsWhite: (playAsWhite: boolean) => void;
    playAsBlack: boolean;
    setPlayAsBlack: (playAsBlack: boolean) => void;
}

const SettingsPanel = ({ 
    loading,
    fen, 
    onFenChange, 
    onFlip,
    onMovePop,
    onResetGame,
    rollouts,
    setRollouts,
    playAsWhite,
    setPlayAsWhite,
    playAsBlack,
    setPlayAsBlack,
}: SettingsPanelProps) => {

    const onFenInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (loading) return;
        onFenChange(e.target.value);
    }

    const onRolloutsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const rollouts = parseInt(e.target.value);

        if (loading) return;
        if (isNaN(rollouts)) return;
        if (rollouts < 0) return;

        setRollouts(rollouts);
    }

    return (
        <div className="SettingsPanel">
            <div className="MainSettingsContainer">
                <div className="PlayAsContainer">
                    <div className="PlayAsLabelsContainer">
                        <div className="PlayAsWhiteLabel">Play as white</div>
                        <div className="PlayAsBlackLabel">Play as black</div>
                    </div>
                    <div className="PlayAsCheckboxContainer">
                        <input className="PlayAsWhiteCheckbox" type="checkbox" checked={playAsWhite} onChange={() => setPlayAsWhite(!playAsWhite)}/>
                        <input className="PlayAsBlackCheckbox" type="checkbox" checked={playAsBlack} onChange={() => setPlayAsBlack(!playAsBlack)}/>
                    </div>
                </div>
                <div className="RolloutsInputContainer">
                    <div className="RolloutsLabel">MCTS rollouts</div>
                    <input className="RolloutsInput" type="number" value={rollouts} onChange={onRolloutsChange}/>
                </div>
            </div>
            <div className="FenAndButtonsContainer">
                <button className="FlipButton" onClick={onFlip}>Flip</button>
                <button className="ResetGameButton" onClick={onResetGame}>Reset</button>
                <button className="PopMoveButton" onClick={onMovePop}>Pop</button>
                <input className="FenInput" type="text" value={fen} onChange={onFenInputChange}/>
            </div>
        </div>
    );
}

export default SettingsPanel;
