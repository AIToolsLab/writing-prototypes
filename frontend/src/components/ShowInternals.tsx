
import { useState } from 'react';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import { getLogprobs } from '../api';
import { Terminal, RefreshCw, BarChart2 } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

// Define specific shape if possible, otherwise use loose typing for now
interface LogprobEntry {
  token: string;
  logprob: number;
  top_logprobs?: Record<string, number>;
}

export default function ShowInternals() {
  const [messages, setMessages] = useState<Message[]>([{ role: 'user', content: '' }]);
  const [logprobs, setLogprobs] = useState<LogprobEntry[]>([]);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const lastMessageIndex = messages.length - 1;
  const activeMessage = messages[lastMessageIndex];

  const handleUpdateActiveMessage = (text: string) => {
    setMessages(prev => {
      const newMessages = [...prev];
      newMessages[lastMessageIndex] = {
        ...newMessages[lastMessageIndex],
        content: text
      };
      return newMessages;
    });
  };

  const handleRewind = (index: number) => {
    setMessages(prev => prev.slice(0, index + 1));
    setLogprobs([]); // Clear analysis on edit
    setSelectedTokenIndex(null);
  };

  const handleSend = () => {
    const nextRole = activeMessage.role === 'user' ? 'assistant' : 'user';
    setMessages(prev => [...prev, { role: nextRole, content: '' }]);
  };

  const fetchLogprobs = async () => {
    setIsLoading(true);
    try {
      // Mock result shape for now, real api should return array of logprob entries
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result: any = await getLogprobs(messages);
      setLogprobs(result || []);
    } catch (e) {
      console.error(e);
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-4 md:p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight text-gray-900 flex items-center">
            <Terminal className="mr-2" size={24} /> Model Internals
        </h2>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Chat / Input Column */}
        <div className="lg:col-span-1 space-y-4">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 space-y-4">
                <h3 className="font-semibold text-gray-900">Conversation Context</h3>
                <div className="space-y-3 max-h-[400px] overflow-y-auto">
                    {messages.slice(0, -1).map((m, i) => (
                    <div key={i} className={`p-3 rounded-lg text-sm border ${m.role === 'user' ? 'bg-blue-50 border-blue-100' : 'bg-gray-50 border-gray-100'}`}>
                        <div className="flex justify-between items-center mb-1">
                            <span className="font-xs uppercase text-gray-400 font-bold">{m.role}</span>
                            <button onClick={() => handleRewind(i)} className="text-xs text-blue-600 hover:underline">Edit</button>
                        </div>
                        <p className="whitespace-pre-wrap">{m.content}</p>
                    </div>
                    ))}
                </div>

                <div className="pt-4 border-t border-gray-100">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        {activeMessage.role === 'user' ? "Current User Message" : "Assistant Response"}
                    </label>
                    <textarea
                        className="w-full rounded-md border border-gray-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent min-h-[120px]"
                        value={activeMessage.content}
                        onChange={e => handleUpdateActiveMessage(e.target.value)}
                        placeholder="Type here..."
                    />
                    <div className="mt-3 flex gap-2">
                        <button 
                            onClick={handleSend}
                            className="flex-1 px-3 py-2 bg-white border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 shadow-sm"
                        >
                            Next Turn
                        </button>
                        <button 
                            onClick={fetchLogprobs}
                            disabled={isLoading}
                            className="flex-1 px-3 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 shadow-sm flex items-center justify-center disabled:opacity-50"
                        >
                            {isLoading ? <RefreshCw size={14} className="animate-spin mr-1"/> : <BarChart2 size={14} className="mr-1"/>}
                            Analyze
                        </button>
                    </div>
                </div>
            </div>
        </div>

        {/* Visualization Column */}
        <div className="lg:col-span-2 space-y-4">
             {/* Token View */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 min-h-[200px]">
                <h3 className="font-semibold text-gray-900 mb-4">Token Analysis</h3>
                {logprobs.length === 0 ? (
                    <div className="text-center text-gray-400 py-12">
                        <p>No analysis data available. Click "Analyze" to see token probabilities.</p>
                    </div>
                ) : (
                    <div className="flex flex-wrap gap-1 leading-relaxed">
                        {logprobs.map((entry, i) => (
                        <span 
                            key={i}
                            className={`cursor-pointer px-1 py-0.5 rounded border transition-colors ${
                                selectedTokenIndex === i 
                                ? 'bg-blue-100 border-blue-300 ring-2 ring-blue-500 ring-opacity-50' 
                                : 'bg-white border-gray-200 hover:bg-gray-50'
                            }`}
                            onClick={() => setSelectedTokenIndex(i)}
                        >
                            {entry.token ? entry.token.replace('\n', '↵') : '[_]'}
                        </span>
                        ))}
                    </div>
                )}
            </div>
            
            {/* Details View */}
            {selectedTokenIndex !== null && logprobs[selectedTokenIndex] && (
                <div className="bg-gray-50 rounded-xl border border-gray-200 p-6 animate-in slide-in-from-top-2">
                    <div className="flex justify-between items-start mb-4">
                        <h4 className="font-mono text-lg font-bold text-gray-900">
                            Token: <span className="bg-white px-2 py-1 rounded border border-gray-200">{JSON.stringify(logprobs[selectedTokenIndex].token)}</span>
                        </h4>
                        <span className="text-sm text-gray-500">Logprob: {logprobs[selectedTokenIndex].logprob?.toFixed(4)}</span>
                    </div>
                    
                    {logprobs[selectedTokenIndex].top_logprobs && (
                         <div>
                            <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Alternative Predictions</h5>
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                                {Object.entries(logprobs[selectedTokenIndex].top_logprobs || {})
                                    .sort(([, a], [, b]) => b - a)
                                    .slice(0, 10) // Top 10
                                    .map(([token, score]) => (
                                    <div key={token} className="bg-white p-2 rounded border border-gray-200 text-sm flex justify-between items-center">
                                        <span className="font-mono text-gray-700 truncate mr-2">{token.replace('\n', '↵')}</span>
                                        <span className="text-xs text-gray-400 font-mono">{score.toFixed(2)}</span>
                                    </div>
                                ))}
                            </div>
                         </div>
                    )}
                </div>
            )}
        </div>
      </div>
    </div>
  );
}
