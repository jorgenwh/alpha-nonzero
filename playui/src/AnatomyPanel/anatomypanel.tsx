import './anatomypanel.css';

interface AnatomyPanelProps {
    evaluate: () => void;
    value: string;
    top5: string[][] | undefined;
}

const buildPolicyIndicatorMoveList = (top5: string[][] | undefined) => {
    if (top5 === undefined) return [];

    return top5.map((k) => {
        const move = k[0];

        return (
            <div className="PolicyIndicatorMove">{move}</div>
        );
    });
}

const buildPolicyIndicatorValueList = (top5: string[][] | undefined) => {
    if (top5 === undefined) return [];

    return top5.map((k) => {
        const value = k[1];
        const displayValue = Number.parseFloat(value).toFixed(3);

        return (
            <div className="PolicyIndicatorValue">{displayValue}</div>
        );
    });
}

const AnatomyPanel = ({ evaluate, value, top5 }: AnatomyPanelProps) => {
    const forceEvaluate = () => { evaluate(); }

    const displayedPolicy = (top5 === undefined) ? 'N/A' : undefined;
    const policyIndicatorMoveList = buildPolicyIndicatorMoveList(top5);
    const policyIndicatorValueList = buildPolicyIndicatorValueList(top5);

    return (
        <div className="AnatomyPanel">
            <h1 className="AnatomyPanelTitle" >Anatomy</h1>
            <button className="EvaluateButton" onClick={forceEvaluate}>Force evaluate</button>
            <div className="ValueIndicator">Value: {value}</div>
            <div className="PolicyIndicatorTitle">Policy: {displayedPolicy}</div>
            <div className="PolicyIndicator">
                <div className="PolicyIndicatorMoveList">
                    {policyIndicatorMoveList}
                </div>
                <div className="PolicyIndicatorValueList">
                    {policyIndicatorValueList}
                </div>
            </div>
        </div>
    );
}

export default AnatomyPanel;
