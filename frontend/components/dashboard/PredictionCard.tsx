'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { PredictionOutput } from '@/lib/api';
import { formatPercentage, formatRelativeTime } from '@/lib/utils';

interface PredictionCardProps {
  prediction: PredictionOutput | null;
  isLoading?: boolean;
}

export function PredictionCard({ prediction, isLoading }: PredictionCardProps) {
  if (isLoading) {
    return (
      <Card className="border-blue-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-600 animate-pulse" />
            Loading Prediction...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32 flex items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Prediction</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500">Select a ticker to see predictions</p>
        </CardContent>
      </Card>
    );
  }

  const isUp = prediction.prediction === 1;
  const confidence = prediction.probability * 100;

  return (
    <Card className={`border-2 ${isUp ? 'border-green-400' : 'border-red-400'}`}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Prediction</span>
          <Badge variant={isUp ? 'success' : 'error'}>
            {prediction.model_version}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Main Prediction */}
          <div className="flex items-center justify-center py-6">
            <div className="text-center">
              <div className={`flex items-center justify-center gap-3 ${isUp ? 'text-green-600' : 'text-red-600'}`}>
                {isUp ? (
                  <TrendingUp className="h-16 w-16" />
                ) : (
                  <TrendingDown className="h-16 w-16" />
                )}
                <span className="text-6xl font-bold">{isUp ? 'UP' : 'DOWN'}</span>
              </div>
            </div>
          </div>

          {/* Confidence Bar */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-600">Confidence</span>
              <span className="font-semibold">{formatPercentage(prediction.probability)}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${isUp ? 'bg-green-500' : 'bg-red-500'}`}
                style={{ width: `${confidence}%` }}
              />
            </div>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-4 text-sm border-t pt-4">
            <div>
              <p className="text-gray-500">Processed</p>
              <p className="font-medium">{formatRelativeTime(prediction.timestamp)}</p>
            </div>
            <div>
              <p className="text-gray-500">Latency</p>
              <p className="font-medium">{prediction.processing_time_ms.toFixed(2)}ms</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
