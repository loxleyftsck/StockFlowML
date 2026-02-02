'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Search, RefreshCw } from 'lucide-react';
import { useState } from 'react';

interface UserControlsProps {
  onPredict: (ticker: string) => void;
  isLoading?: boolean;
}

export function UserControls({ onPredict, isLoading }: UserControlsProps) {
  const [ticker, setTicker] = useState('BBCA.JK');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(ticker);
  };

  const popularTickers = ['BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK'];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5 text-blue-600" />
          Ticker Selection
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Ticker Input */}
          <div>
            <label htmlFor="ticker" className="block text-sm font-medium text-gray-700 mb-2">
              Stock Ticker
            </label>
            <input
              id="ticker"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g., BBCA.JK"
              disabled={isLoading}
            />
          </div>

          {/* Popular Tickers */}
          <div>
            <p className="text-sm text-gray-600 mb-2">Popular Tickers:</p>
            <div className="flex flex-wrap gap-2">
              {popularTickers.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTicker(t)}
                  className={`px-3 py-1 text-sm rounded-full border transition-colors ${
                    ticker === t
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white text-gray-700 border-gray-300 hover:border-blue-500'
                  }`}
                  disabled={isLoading}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                Fetching Prediction...
              </>
            ) : (
              <>
                <Search className="h-4 w-4" />
                Get Prediction
              </>
            )}
          </button>
        </form>
      </CardContent>
    </Card>
  );
}
