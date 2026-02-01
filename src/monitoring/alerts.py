"""
Alert System Module

Sends alerts via Discord webhooks and email when drift or performance
degradation is detected.
"""

import requests
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import os


class AlertSystem:
    """
    Alert system for monitoring notifications.
    
    Supports Discord webhooks and can be extended for email alerts.
    """
    
    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize alert system.
        
        Args:
            discord_webhook_url: Discord webhook URL (or set DISCORD_WEBHOOK_URL env var)
            enabled: Whether alerts are enabled
        """
        self.enabled = enabled
        self.discord_webhook_url = discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        
    def send_discord_alert(
        self,
        title: str,
        message: str,
        color: str = "warning",
        fields: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """
        Send alert to Discord via webhook.
        
        Args:
            title: Alert title
            message: Alert message/description
            color: Alert color - 'success', 'warning', 'error', 'info'
            fields: List of {"name": "Field Name", "value": "Field Value"} dicts
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            print("[ALERT] Alerts disabled - skipping")
            return False
            
        if not self.discord_webhook_url:
            print("[ALERT] Discord webhook URL not configured")
            return False
        
        # Color mapping
        color_codes = {
            "success": 0x00FF00,  # Green
            "warning": 0xFFA500,  # Orange
            "error": 0xFF0000,    # Red
            "info": 0x0099FF,     # Blue
        }
        
        # Build embed
        embed = {
            "title": title,
            "description": message,
            "color": color_codes.get(color, color_codes["info"]),
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "StockFlowML Monitoring"
            }
        }
        
        # Add fields if provided
        if fields:
            embed["fields"] = [
                {
                    "name": field["name"],
                    "value": field["value"],
                    "inline": field.get("inline", True)
                }
                for field in fields
            ]
        
        # Send to Discord
        payload = {
            "embeds": [embed]
        }
        
        try:
            response = requests.post(
                self.discord_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                print(f"[ALERT] Discord alert sent: {title}")
                return True
            else:
                print(f"[ALERT] Discord alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[ALERT] Discord alert error: {e}")
            return False
    
    def send_drift_alert(
        self,
        ticker: str,
        drift_summary: Dict[str, Any],
        report_path: Optional[Path] = None
    ) -> bool:
        """
        Send drift detection alert.
        
        Args:
            ticker: Stock ticker symbol
            drift_summary: Drift summary from DriftDetector
            report_path: Optional path to HTML report
            
        Returns:
            True if sent successfully
        """
        data_drift = drift_summary.get("data_drift", {})
        target_drift = drift_summary.get("target_drift", {})
        
        # Determine severity
        has_data_drift = data_drift.get("dataset_drift_detected", False)
        has_target_drift = target_drift.get("target_drift_detected", False)
        
        if has_data_drift or has_target_drift:
            color = "warning"
            title = f"DRIFT DETECTED - {ticker}"
        else:
            color = "success"
            title = f"No Drift Detected - {ticker}"
        
        # Build message
        message = "**Drift Detection Results**\n\n"
        
        if has_data_drift:
            drift_share = data_drift.get("drift_share", 0) * 100
            num_drifted = data_drift.get("num_drifted_columns", 0)
            message += f"- Data Drift: YES ({drift_share:.1f}% of features, {num_drifted} columns)\n"
        else:
            message += "- Data Drift: NO\n"
        
        if has_target_drift:
            drift_score = target_drift.get("target_drift_score", 0)
            message += f"- Target Drift: YES (score: {drift_score:.3f})\n"
        else:
            message += "- Target Drift: NO\n"
        
        # Add fields
        fields = [
            {
                "name": "Ticker",
                "value": ticker,
                "inline": True
            },
            {
                "name": "Timestamp",
                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "inline": True
            }
        ]
        
        if has_data_drift or has_target_drift:
            fields.append({
                "name": "Action Required",
                "value": "Review drift report and consider model retraining",
                "inline": False
            })
        
        if report_path and report_path.exists():
            fields.append({
                "name": "Report",
                "value": f"Check `{report_path.name}` for details",
                "inline": False
            })
        
        return self.send_discord_alert(
            title=title,
            message=message,
            color=color,
            fields=fields
        )
    
    def send_performance_alert(
        self,
        ticker: str,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        degradation_pct: float
    ) -> bool:
        """
        Send model performance degradation alert.
        
        Args:
            ticker: Stock ticker
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics  
            degradation_pct: Percentage degradation
            
        Returns:
            True if sent successfully
        """
        title = f"PERFORMANCE DEGRADATION - {ticker}"
        
        message = f"**Model performance has degraded by {degradation_pct:.1f}%**\n\n"
        message += "Current vs Baseline:\n"
        
        fields = []
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            current = current_metrics.get(metric_name, 0)
            baseline = baseline_metrics.get(metric_name, 0)
            
            fields.append({
                "name": metric_name.capitalize(),
                "value": f"{current:.3f} (was {baseline:.3f})",
                "inline": True
            })
        
        fields.append({
            "name": "Action Required",
            "value": "Model should be retrained with recent data",
            "inline": False
        })
        
        return self.send_discord_alert(
            title=title,
            message=message,
            color="error",
            fields=fields
        )
    
    def send_training_complete_alert(
        self,
        ticker: str,
        metrics: Dict[str, float],
        training_time: float
    ) -> bool:
        """
        Send notification when model training completes.
        
        Args:
            ticker: Stock ticker
            metrics: Training metrics
            training_time: Training duration in seconds
            
        Returns:
            True if sent successfully
        """
        title = f"Training Complete - {ticker}"
        
        message = f"**Model training finished successfully**\n\n"
        message += f"Duration: {training_time:.1f}s"
        
        fields = [
            {
                "name": "Accuracy",
                "value": f"{metrics.get('accuracy', 0):.3f}",
                "inline": True
            },
            {
                "name": "F1-Score",
                "value": f"{metrics.get('f1', 0):.3f}",
                "inline": True
            },
            {
                "name": "Ticker",
                "value": ticker,
                "inline": True
            }
        ]
        
        return self.send_discord_alert(
            title=title,
            message=message,
            color="success",
            fields=fields
        )
    
    def test_connection(self) -> bool:
        """
        Test Discord webhook connection.
        
        Returns:
            True if connection successful
        """
        return self.send_discord_alert(
            title="Test Alert",
            message="StockFlowML alert system is working correctly!",
            color="info",
            fields=[
                {
                    "name": "Status",
                    "value": "Connection OK",
                    "inline": False
                }
            ]
        )


# Convenience function
def send_drift_alert(
    ticker: str,
    drift_summary: Dict[str, Any],
    webhook_url: Optional[str] = None
) -> bool:
    """
    Quick function to send drift alert.
    
    Args:
        ticker: Stock ticker
        drift_summary: Drift summary from DriftDetector
        webhook_url: Discord webhook URL (optional, uses env var if not provided)
        
    Returns:
        True if sent successfully
    """
    alert_system = AlertSystem(discord_webhook_url=webhook_url)
    return alert_system.send_drift_alert(ticker, drift_summary)


if __name__ == "__main__":
    # Example usage
    print("Alert System - Example Usage")
    print("-" * 50)
    
    # Check if webhook is configured
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        print("Set DISCORD_WEBHOOK_URL environment variable to test alerts")
        print("\nExample:")
        print('  $env:DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."')
        print("  python -m src.monitoring.alerts")
    else:
        alert_system = AlertSystem()
        print("Testing Discord webhook connection...")
        
        if alert_system.test_connection():
            print("SUCCESS - Alert sent to Discord!")
        else:
            print("FAILED - Check webhook URL and connection")
