import React from 'react';

const ProbabilityBar = ({ homeProb, drawProb, awayProb }) => {
  const formatProb = (prob) => (prob * 100).toFixed(1) + '%';
  
  return (
    <div className="w-full flex h-2 rounded-full overflow-hidden bg-dark-700/50 my-4 border border-dark-600/50">
      <div 
        className="h-full bg-neon-green/80 transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(16,185,129,0.5)]"
        style={{ width: `${homeProb * 100}%` }}
        title={`Home: ${formatProb(homeProb)}`}
      />
      <div 
        className="h-full bg-yellow-500/80 transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(234,179,8,0.5)]"
        style={{ width: `${drawProb * 100}%` }}
        title={`Draw: ${formatProb(drawProb)}`}
      />
      <div 
        className="h-full bg-neon-red/80 transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(239,68,68,0.5)]"
        style={{ width: `${awayProb * 100}%` }}
        title={`Away: ${formatProb(awayProb)}`}
      />
    </div>
  );
};

export default ProbabilityBar;
