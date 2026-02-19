#!/usr/bin/env python3
"""
Event Market Scanner
====================
Collects top 1000 events (by liquidity) from Polymarket and Kalshi,
including prices for all possible outcomes and event liquidity.

Usage:
    python event_market_scanner.py                    # Fetch from both platforms
    python event_market_scanner.py --polymarket       # Polymarket only
    python event_market_scanner.py --kalshi           # Kalshi only
    python event_market_scanner.py --limit 100        # Top 100 instead of 1000
    python event_market_scanner.py --export data.csv  # Export to CSV
    python event_market_scanner.py --detail 5         # Show details of event #5
    python event_market_scanner.py --top 20           # Display top 20 in table
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import requests

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()
    RICH = True
except ImportError:
    RICH = False


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Outcome:
    """A single outcome (e.g. 'Yes' at 0.65)."""
    name: str
    price: float  # 0.0 to 1.0


@dataclass
class Market:
    """A single market within an event (e.g. 'Will X happen?')."""
    question: str
    outcomes: list
    liquidity: float = 0.0


@dataclass
class Event:
    """An event containing one or more markets."""
    title: str
    markets: list
    liquidity: float = 0.0
    volume: float = 0.0
    source: str = ""
    url: str = ""
    category: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _log(msg: str):
    if RICH:
        console.print(f"  [yellow]{msg}[/yellow]")
    else:
        print(f"  {msg}", file=sys.stderr)


def _info(msg: str):
    if RICH:
        console.print(f"  [dim]{msg}[/dim]")
    else:
        print(f"  {msg}", file=sys.stderr)


def fmt_usd(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"${val / 1_000:,.1f}K"
    return f"${val:,.0f}"


def fmt_num(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:,.1f}K"
    return f"{val:,.0f}"


def outcomes_str(outcomes: list, max_show: int = 5) -> str:
    parts = [f"{o.name}: {o.price:.1%}" for o in outcomes[:max_show]]
    text = " | ".join(parts)
    if len(outcomes) > max_show:
        text += f" (+{len(outcomes) - max_show} more)"
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Polymarket Collector
# ──────────────────────────────────────────────────────────────────────────────


class PolymarketCollector:
    """Collects event data from the Polymarket Gamma API."""

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "EventMarketScanner/1.0",
        })

    def collect(self, limit: int = 1000) -> list:
        """Fetch top events sorted by liquidity."""
        raw_events = []
        page_size = 100
        offset = 0

        while len(raw_events) < limit:
            params = {
                "limit": min(page_size, limit - len(raw_events)),
                "offset": offset,
                "order": "liquidity",
                "ascending": "false",
                "closed": "false",
                "active": "true",
            }
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/events", params=params, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                _log(f"Polymarket API error (offset={offset}): {e}")
                break

            # Handle both list response and dict-wrapped response
            if isinstance(data, dict):
                batch = data.get("events", data.get("data", []))
            elif isinstance(data, list):
                batch = data
            else:
                break

            if not batch:
                break

            raw_events.extend(batch)
            offset += page_size
            _info(f"Polymarket: fetched {len(raw_events)} events...")
            time.sleep(0.2)

        events = []
        for item in raw_events[:limit]:
            try:
                events.append(self._parse(item))
            except Exception:
                pass

        return events

    def _parse(self, data: dict) -> Event:
        markets = []
        total_liq = 0.0
        total_vol = 0.0

        for m in data.get("markets", []):
            # outcomes and prices can be JSON strings or lists
            out_names = m.get("outcomes", "[]")
            out_prices = m.get("outcomePrices", "[]")

            if isinstance(out_names, str):
                out_names = json.loads(out_names)
            if isinstance(out_prices, str):
                out_prices = json.loads(out_prices)

            outcomes = [
                Outcome(name=str(n), price=float(p))
                for n, p in zip(out_names, out_prices)
            ]

            if not outcomes:
                continue  # skip markets with no outcomes

            liq = float(m.get("liquidity") or 0)
            vol = float(m.get("volume") or 0)
            total_liq += liq
            total_vol += vol

            markets.append(Market(
                question=m.get("question", ""),
                outcomes=outcomes,
                liquidity=liq,
            ))

        slug = data.get("slug", "")
        return Event(
            title=data.get("title", "Unknown"),
            markets=markets,
            liquidity=total_liq,
            volume=total_vol,
            source="Polymarket",
            url=f"https://polymarket.com/event/{slug}" if slug else "",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Kalshi Collector
# ──────────────────────────────────────────────────────────────────────────────


class KalshiCollector:
    """Collects event data from the Kalshi Trading API v2."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "EventMarketScanner/1.0",
        })

    def collect(self, limit: int = 1000) -> list:
        """Fetch events from Kalshi, sorted by liquidity."""
        raw_events = []
        cursor = None
        max_pages = 100

        for page in range(max_pages):
            params = {
                "limit": 200,
                "status": "open",
                "with_nested_markets": "true",
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/events", params=params, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                _log(f"Kalshi API error (page={page}): {e}")
                break

            batch = data.get("events", [])
            if not batch:
                break

            raw_events.extend(batch)
            cursor = data.get("cursor")
            _info(f"Kalshi: fetched {len(raw_events)} events...")

            if not cursor:
                break
            # Stop fetching once we have 3x the requested limit
            # (buffer for sorting since Kalshi doesn't sort server-side)
            if len(raw_events) >= limit * 3:
                _info(f"Kalshi: reached fetch cap ({len(raw_events)} events)")
                break
            time.sleep(0.2)

        # Parse all, then sort by liquidity descending
        events = []
        for item in raw_events:
            try:
                events.append(self._parse(item))
            except Exception:
                pass

        events.sort(key=lambda e: e.liquidity, reverse=True)
        return events[:limit]

    def _parse(self, data: dict) -> Event:
        is_exclusive = data.get("mutually_exclusive", False)
        raw_markets = data.get("markets", [])

        total_liq = 0.0
        total_vol = 0

        if is_exclusive and len(raw_markets) > 1:
            # Mutually exclusive: treat entire event as one market with
            # each sub-market representing a single named outcome.
            outcomes = []
            for m in raw_markets:
                name = m.get("yes_sub_title") or m.get("subtitle") or m.get("ticker", "")
                yes_price = self._mid_price(m)
                liq = self._parse_dollars(m.get("liquidity_dollars", "0"))
                vol = m.get("volume") or 0
                total_liq += liq
                total_vol += vol
                outcomes.append(Outcome(name, round(yes_price, 4)))

            # Sort outcomes by price descending
            outcomes = [o for o in outcomes if o.name]  # drop unnamed
            outcomes.sort(key=lambda o: o.price, reverse=True)

            if not outcomes:
                markets = []
            else:
                event_title = data.get("title", "Unknown")
                markets = [Market(
                question=event_title,
                outcomes=outcomes,
                liquidity=total_liq,
            )]
        else:
            # Independent markets — each keeps its own Yes/No outcomes
            markets = []
            for m in raw_markets:
                yes_price = self._mid_price(m)
                no_price = 1.0 - yes_price
                liq = self._parse_dollars(m.get("liquidity_dollars", "0"))
                vol = m.get("volume") or 0
                total_liq += liq
                total_vol += vol

                title = (m.get("yes_sub_title")
                         or m.get("subtitle")
                         or m.get("title")
                         or m.get("ticker", ""))

                outcomes = [
                    Outcome("Yes", round(yes_price, 4)),
                    Outcome("No", round(no_price, 4)),
                ]
                markets.append(Market(question=title, outcomes=outcomes, liquidity=liq))

        ticker = data.get("event_ticker", "")
        return Event(
            title=data.get("title", "Unknown"),
            markets=markets,
            liquidity=total_liq,
            volume=total_vol,
            source="Kalshi",
            url=f"https://kalshi.com/markets/{ticker}" if ticker else "",
            category=data.get("category", ""),
        )

    @staticmethod
    def _mid_price(m: dict) -> float:
        """Return midpoint of bid/ask in 0-1 range; fallback to last price."""
        yes_bid = (m.get("yes_bid") or 0) / 100
        yes_ask = (m.get("yes_ask") or 0) / 100
        last = (m.get("last_price") or 0) / 100
        if yes_bid and yes_ask:
            return (yes_bid + yes_ask) / 2
        return last

    @staticmethod
    def _parse_dollars(val) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Display — Rich
# ──────────────────────────────────────────────────────────────────────────────


def display_rich(events: list, title: str, source: str):
    """Render an events table using the rich library."""
    console.print()

    header = (
        f"[bold]{title}[/bold]\n"
        f"[dim]Fetched {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
    )
    console.print(Panel(header, border_style="cyan", expand=False))
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_lines=True,
        pad_edge=True,
        expand=False,
    )
    table.add_column("#", style="dim", width=5, justify="right")
    table.add_column("Event", style="bold white", max_width=48, no_wrap=False)

    table.add_column("Liquidity ($)", justify="right", style="green")
    table.add_column("Volume ($)", justify="right", style="cyan")

    table.add_column("Mkts", justify="center", width=5)
    table.add_column("Outcome Prices", max_width=64, no_wrap=False)

    for i, event in enumerate(events, 1):
        # Build outcome display
        if len(event.markets) == 0:
            out_text = "[dim]No markets[/dim]"
        elif len(event.markets) == 1:
            out_text = _rich_outcomes_single(event.markets[0])
        elif len(event.markets) <= 4:
            out_text = _rich_outcomes_multi(event.markets, show=4)
        else:
            out_text = _rich_outcomes_multi(event.markets[:3], show=3)
            out_text += f"\n[dim]  … +{len(event.markets) - 3} more markets[/dim]"

        liq = fmt_usd(event.liquidity)
        vol = fmt_usd(event.volume)

        table.add_row(str(i), event.title, liq, vol, str(len(event.markets)), out_text)

    console.print(table)


def _rich_outcomes_single(market: Market) -> str:
    parts = []
    for o in market.outcomes:
        color = "green" if o.price >= 0.5 else ("red" if o.price < 0.15 else "yellow")
        parts.append(f"[{color}]{o.name}: {o.price:.1%}[/{color}]")
    return " | ".join(parts)


def _rich_outcomes_multi(markets: list, show: int = 4) -> str:
    lines = []
    for m in markets[:show]:
        q = (m.question[:35] + "…") if len(m.question) > 35 else m.question
        prices = ", ".join(f"{o.name}: {o.price:.0%}" for o in m.outcomes[:4])
        lines.append(f"[dim]{q}:[/dim] {prices}")
    return "\n".join(lines)


def detail_rich(event: Event, rank: int):
    """Show full detail of one event."""
    console.print()
    console.print(Panel(
        f"[bold]{event.title}[/bold]\n"
        f"Source: {event.source}  |  Liquidity: {fmt_usd(event.liquidity)}  |  "
        f"Volume: {fmt_usd(event.volume)}\n"
        f"URL: [link={event.url}]{event.url}[/link]",
        title=f"[bold]Event #{rank}[/bold]",
        border_style="green",
    ))

    for idx, market in enumerate(event.markets, 1):
        t = Table(
            title=f"Market {idx}: {market.question}",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            title_style="bold",
        )
        t.add_column("Outcome", style="bold")
        t.add_column("Price ($)", justify="right")
        t.add_column("Implied Probability", justify="right")
        t.add_column("Market Liquidity", justify="right")

        for o in market.outcomes:
            clr = "green" if o.price >= 0.5 else ("red" if o.price < 0.15 else "yellow")
            t.add_row(
                o.name,
                f"${o.price:.4f}",
                f"[{clr}]{o.price:.1%}[/{clr}]",
                fmt_usd(market.liquidity),
            )

        console.print(t)
    console.print()


# ──────────────────────────────────────────────────────────────────────────────
# Display — Plain text fallback
# ──────────────────────────────────────────────────────────────────────────────


def display_plain(events: list, title: str, source: str):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"  Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 90}\n")

    for i, event in enumerate(events, 1):
        liq = fmt_usd(event.liquidity)
        vol = fmt_usd(event.volume)
        print(f"  {i:>4}. {event.title[:65]}")
        print(f"        Liquidity: {liq}  |  Volume: {vol}  |  Markets: {len(event.markets)}")

        for m in event.markets[:3]:
            q = (m.question[:42] + "…") if len(m.question) > 42 else m.question
            prices = ", ".join(f"{o.name}: {o.price:.1%}" for o in m.outcomes[:5])
            print(f"        └─ {q}: {prices}")

        if len(event.markets) > 3:
            print(f"        └─ … +{len(event.markets) - 3} more markets")
        print()


def detail_plain(event: Event, rank: int):
    print(f"\n{'─' * 70}")
    print(f"  Event #{rank}: {event.title}")
    print(f"  Source: {event.source}  |  Liquidity: {fmt_usd(event.liquidity)}")
    print(f"  Volume: {fmt_usd(event.volume)}  |  URL: {event.url}")
    print(f"{'─' * 70}")

    for idx, m in enumerate(event.markets, 1):
        print(f"\n  Market {idx}: {m.question}")
        print(f"  {'Outcome':<25} {'Price':>10} {'Probability':>14}")
        print(f"  {'─' * 50}")
        for o in m.outcomes:
            print(f"  {o.name:<25} ${o.price:<9.4f} {o.price:>13.1%}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Display router
# ──────────────────────────────────────────────────────────────────────────────


def display_events(events: list, title: str, source: str):
    if RICH:
        display_rich(events, title, source)
    else:
        display_plain(events, title, source)


def display_detail(event: Event, rank: int):
    if RICH:
        detail_rich(event, rank)
    else:
        detail_plain(event, rank)


# ──────────────────────────────────────────────────────────────────────────────
# CSV Export
# ──────────────────────────────────────────────────────────────────────────────


def export_csv(events: list, filepath: str):
    """Write all events / markets / outcomes to a flat CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Source", "Event", "Event Liquidity", "Event Volume",
            "Market", "Market Liquidity", "Outcome", "Price", "URL",
        ])
        for rank, event in enumerate(events, 1):
            for market in event.markets:
                for outcome in market.outcomes:
                    writer.writerow([
                        rank,
                        event.source,
                        event.title,
                        f"{event.liquidity:.2f}",
                        f"{event.volume:.2f}",
                        market.question,
                        f"{market.liquidity:.2f}",
                        outcome.name,
                        f"{outcome.price:.4f}",
                        event.url,
                    ])
    _log(f"Exported {len(events)} events → {filepath}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Event Market Scanner — top prediction-market events from Polymarket & Kalshi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python event_market_scanner.py                      # Both platforms, top 1000
  python event_market_scanner.py --polymarket          # Polymarket only
  python event_market_scanner.py --kalshi              # Kalshi only
  python event_market_scanner.py --limit 200           # Top 200 per platform
  python event_market_scanner.py --top 20              # Show 20 rows in table
  python event_market_scanner.py --export results.csv  # Save to CSV
  python event_market_scanner.py --detail 3            # Full detail for event #3
        """,
    )
    parser.add_argument("--polymarket", action="store_true",
                        help="Fetch from Polymarket only")
    parser.add_argument("--kalshi", action="store_true",
                        help="Fetch from Kalshi only")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max events to collect per platform (default: 1000)")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of rows to display (default: 50, 0 = all)")
    parser.add_argument("--export", metavar="FILE",
                        help="Export results to a CSV file")
    parser.add_argument("--detail", type=int, metavar="N",
                        help="Show detailed breakdown of event #N")

    args = parser.parse_args()

    # If neither flag → fetch both
    fetch_poly = args.polymarket or (not args.polymarket and not args.kalshi)
    fetch_kalshi = args.kalshi or (not args.polymarket and not args.kalshi)

    # Banner
    if RICH:
        console.print(Panel(
            "[bold cyan]Event Market Scanner[/bold cyan]\n"
            "[dim]Prediction market data from Polymarket & Kalshi[/dim]",
            expand=False,
        ))
    else:
        print("\n  ── Event Market Scanner ──")
        print("  Prediction market data from Polymarket & Kalshi\n")

    # ── Collect ──────────────────────────────────────────────────────────
    poly_events: list = []
    kalshi_events: list = []

    if fetch_poly:
        _log("Fetching Polymarket events …")
        try:
            poly_events = PolymarketCollector().collect(limit=args.limit)
            _log(f"✓ Polymarket: {len(poly_events)} events collected")
        except Exception as e:
            _log(f"✗ Polymarket failed: {e}")

    if fetch_kalshi:
        _log("Fetching Kalshi events …")
        try:
            kalshi_events = KalshiCollector().collect(limit=args.limit)
            _log(f"✓ Kalshi: {len(kalshi_events)} events collected")
        except Exception as e:
            _log(f"✗ Kalshi failed: {e}")

    # ── Detail view ──────────────────────────────────────────────────────
    all_events = poly_events + kalshi_events

    if args.detail:
        if 1 <= args.detail <= len(all_events):
            display_detail(all_events[args.detail - 1], args.detail)
        else:
            _log(f"Event #{args.detail} not found (valid: 1–{len(all_events)})")
        return

    # ── Table view ───────────────────────────────────────────────────────
    show = args.top if args.top > 0 else None

    if poly_events:
        shown = poly_events[:show] if show else poly_events
        display_events(
            shown,
            f"Polymarket — Top {len(shown)} of {len(poly_events)} Events by Liquidity",
            "Polymarket",
        )

    if kalshi_events:
        shown = kalshi_events[:show] if show else kalshi_events
        display_events(
            shown,
            f"Kalshi — Top {len(shown)} of {len(kalshi_events)} Events by Liquidity",
            "Kalshi",
        )

    # ── Summary ──────────────────────────────────────────────────────────
    if RICH:
        lines = [f"[bold]Total events collected: {len(all_events)}[/bold]"]
        if poly_events:
            total_liq = sum(e.liquidity for e in poly_events)
            lines.append(f"  Polymarket: {len(poly_events)} events — {fmt_usd(total_liq)} total liquidity")
        if kalshi_events:
            total_liq_k = sum(e.liquidity for e in kalshi_events)
            lines.append(f"  Kalshi: {len(kalshi_events)} events — {fmt_usd(total_liq_k)} total liquidity")
        console.print(Panel("\n".join(lines), title="Summary", border_style="blue", expand=False))
    else:
        print(f"\n  Total events: {len(all_events)}")
        if poly_events:
            print(f"  Polymarket: {len(poly_events)} events")
        if kalshi_events:
            print(f"  Kalshi: {len(kalshi_events)} events")

    # ── Export ────────────────────────────────────────────────────────────
    if args.export:
        export_csv(all_events, args.export)

    # ── Tips ──────────────────────────────────────────────────────────────
    if RICH and not args.export:
        console.print(
            "\n[dim]Tips: --export data.csv  |  --detail N  |  "
            "--top N  |  --limit N  |  --polymarket / --kalshi[/dim]\n"
        )


if __name__ == "__main__":
    main()
