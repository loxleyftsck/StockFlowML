'use client';

import { useState } from 'react';
import useSWR from 'swr';
import { PredictionCard } from '@/components/dashboard/PredictionCard';
import { ModelStatus } from '@/components/dashboard/ModelStatus';
import { UserControls } from '@/components/dashboard/UserControls';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { predictTicker, getHealth, PredictionOutput, HealthCheck } from '@/lib/api';
import { BarChart3, FileText, Users, TrendingUp } from 'lucide-react';

export default function DashboardPage() {
  const [prediction, setPrediction] = useState<PredictionOutput | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch health status
  const { data: health } = useSWR<HealthCheck>('health', getHealth, {
    refreshInterval: 30000, // Poll every 30s
  });

  const handlePredict = async (ticker: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await predictTicker({ ticker });
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch prediction');
      console.error('Prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="bg-white border-b shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">StockFlowML Dashboard</h1>
              <p className="text-sm text-gray-500">Production ML Stock Trend Prediction</p>
            </div>
            <div className="flex items-center gap-4">
              {health && (
                <Badge variant={health.status === 'ok' ? 'success' : 'error'}>
                  API: {health.status.toUpperCase()}
                </Badge>
              )}
              {health?.feast_enabled && (
                <Badge variant="info">Feast Enabled</Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar - Technical View */}
          <div className="lg:col-span-3 space-y-6">
            <Card className="border-purple-200">
              <CardHeader>
                <CardTitle className="text-purple-700 flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Technical View
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="bg-purple-50 rounded-lg p-3">
                  <p className="text-sm font-medium text-purple-900 mb-1">Drift Reports</p>
                  <p className="text-xs text-purple-600">View model drift analysis</p>
                </div>
                <div className="bg-purple-50 rounded-lg p-3">
                  <p className="text-sm font-medium text-purple-900 mb-1">Training Logs</p>
                  <p className="text-xs text-purple-600">Access training history</p>
                </div>
                <div className="bg-purple-50 rounded-lg p-3">
                  <p className="text-sm font-medium text-purple-900 mb-1">Model Comparison</p>
                  <p className="text-xs text-purple-600">Compare model versions</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Center - Main Dashboard */}
          <div className="lg:col-span-6 space-y-6">
            {/* User Controls */}
            <UserControls onPredict={handlePredict} isLoading={isLoading} />

            {/* Error Display */}
            {error && (
              <Card className="border-red-200 bg-red-50">
                <CardContent className="pt-6">
                  <p className="text-red-800 text-sm">{error}</p>
                </CardContent>
              </Card>
            )}

            {/* Prediction Display */}
            <PredictionCard prediction={prediction} isLoading={isLoading} />

            {/* Model Status */}
            <ModelStatus 
              modelVersion={health?.version}
              lastTrained={new Date().toISOString()}
            />
          </div>

          {/* Right Sidebar - Business View */}
          <div className="lg:col-span-3 space-y-6">
            <Card className="border-green-200">
              <CardHeader>
                <CardTitle className="text-green-700 flex items-center gap-2">
                  <Users className="h-5 w-5" />
                  Business View
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="bg-green-50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingUp className="h-4 w-4 text-green-600" />
                    <p className="text-sm font-medium text-green-900">Trends</p>
                  </div>
                  <p className="text-xs text-green-600">Market overview & insights</p>
                </div>
                <div className="bg-green-50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <BarChart3 className="h-4 w-4 text-green-600" />
                    <p className="text-sm font-medium text-green-900">Metrics</p>
                  </div>
                  <p className="text-xs text-green-600">Simplified KPIs</p>
                </div>
                <div className="bg-green-50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <FileText className="h-4 w-4 text-green-600" />
                    <p className="text-sm font-medium text-green-900">Summary</p>
                  </div>
                  <p className="text-xs text-green-600">Executive reports</p>
                </div>
              </CardContent>
            </Card>

            {/* Future Features Badge */}
            <Card className="border-gray-200 bg-gray-50">
              <CardHeader>
                <CardTitle className="text-sm">Coming Soon</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="warning">Real-time</Badge>
                  <Badge variant="info">Mobile</Badge>
                  <Badge variant="default">Alerts</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t bg-white mt-12">
        <div className="container mx-auto px-4 py-6 text-center text-sm text-gray-600">
          <p>StockFlowML v1.0 | Production-ready MLOps Pipeline</p>
        </div>
      </footer>
    </div>
  );
}
