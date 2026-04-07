#!/usr/bin/env python3
"""Quick script to check bot status"""

import asyncio
from core.config_loader import get_config
from core.portfolio_tracker import PortfolioTracker

async def main():
    config = get_config()
    tracker = PortfolioTracker(config)
    
    # Get portfolio state (synchronous methods)
    portfolio = tracker.get_portfolio_summary()
    positions = tracker.get_positions()
    
    print("="*60)
    print("PORTFOLIO STATUS") 
    print("="*60)
    print(f"Portfolio Value: ${portfolio['total_value']:.2f}")
    print(f"Cash Available: ${portfolio['cash']:.2f}")
    print(f"Open Positions: {len(positions)}")
    print(f"Max Positions: {config.risk.max_positions}")
    print()
    
    if positions:
        print("OPEN POSITIONS:")
        for pos in positions:
            print(f"  {pos.symbol}: {pos.qty} @ ${float(pos.entry_price):.2f}")
            print(f"    Current: ${float(pos.current_price):.2f} | P&L: ${float(pos.unrealized_pnl):.2f}")
    else:
        print("No open positions")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
