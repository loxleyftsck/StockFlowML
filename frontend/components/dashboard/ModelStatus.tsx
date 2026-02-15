'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { CheckCircle2, Clock, Database, Activity } from 'lucide-react';
import { formatRelativeTime, formatPercentage } from '@/lib/utils';

interface ModelStatusProps {
  modelVersion?: string;
  lastTrained?: string;
  dataFreshness?: string;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
}

export function ModelStatus({ 
  modelVersion = 'v1.0.2',
  lastTrained = new Date().toISOString(),
  dataFreshness = 'Fresh',
  metrics = {
    accuracy: 0.85,
    precision: 0.83,
    recall: 0.87,
    f1: 0.85
  }
}: ModelStatusProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-blue-600" />
          Model Status
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Version & Status */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500 mb-1">Version</p>
              <Badge variant="info">{modelVersion}</Badge>
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Status</p>
              <Badge variant="success" className="flex items-center gap-1 w-fit">
                <CheckCircle2 className="h-3 w-3" />
                Active
              </Badge>
            </div>
          </div>

          {/* Last Trained */}
          <div>
            <p className="text-sm text-gray-500 mb-1 flex items-center gap-1">
              <Clock className="h-4 w-4" />
              Last Trained
            </p>
            <p className="font-medium">{formatRelativeTime(lastTrained)}</p>
          </div>

          {/* Data Freshness */}
          <div>
            <p className="text-sm text-gray-500 mb-1 flex items-center gap-1">
              <Database className="h-4 w-4" />
              Data Freshness
            </p>
            <Badge variant="success">{dataFreshness}</Badge>
          </div>

          {/* Performance Metrics */}
          <div className="border-t pt-4">
            <p className="text-sm font-semibold mb-3">Performance Metrics</p>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Accuracy', value: metrics.accuracy },
                { label: 'Precision', value: metrics.precision },
                { label: 'Recall', value: metrics.recall },
                { label: 'F1-Score', value: metrics.f1 },
              ].map((metric) => (
                <div key={metric.label} className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">{metric.label}</p>
                  <p className="text-lg font-bold text-blue-600">{formatPercentage(metric.value)}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
