export const SIGNALS = [
  {
    id: "signal-001",
    round: 1,
    ticker: "ALP",
    title: "Alpha Desk upgrade after stealth partnership leak",
    description:
      "Two independent channel checks point to a high-margin distribution deal getting signed before earnings.",
    sector: "Technology",
    value: 180,
    isAlpha: true,
  },
  {
    id: "signal-002",
    round: 2,
    ticker: "BRX",
    title: "Broker circular looks flashy but conflicts with supplier filings",
    description:
      "The rumor reads strong, but the filings underneath imply inventory stress and weakening demand.",
    sector: "Consumer",
    value: 140,
    isAlpha: false,
  },
  {
    id: "signal-003",
    round: 3,
    ticker: "CTA",
    title: "Regulatory clearance quietly removes the main overhang",
    description:
      "The company secured the last approval needed to expand into a locked international market.",
    sector: "Healthcare",
    value: 220,
    isAlpha: true,
  },
  {
    id: "signal-004",
    round: 4,
    ticker: "DYN",
    title: "CEO interview boosts sentiment but adds no new fundamentals",
    description:
      "Momentum traders are reacting to a viral clip, yet the actual disclosures stay unchanged.",
    sector: "Media",
    value: 125,
    isAlpha: false,
  },
  {
    id: "signal-005",
    round: 5,
    ticker: "EVO",
    title: "Channel inventory snapped up ahead of product launch",
    description:
      "Sell-through data shows demand moving meaningfully ahead of consensus estimates.",
    sector: "Retail",
    value: 260,
    isAlpha: true,
  },
  {
    id: "signal-006",
    round: 6,
    ticker: "FLS",
    title: "Anonymous forum post claims turnaround, but lenders stay silent",
    description:
      "There is no confirming data from lenders, customers, or management to support the claim.",
    sector: "Industrials",
    value: 150,
    isAlpha: false,
  },
  {
    id: "signal-007",
    round: 7,
    ticker: "GRA",
    title: "Procurement win appears in government ledger before PR release",
    description:
      "A multi-year contract has been logged publicly even though the company has not announced it yet.",
    sector: "Infrastructure",
    value: 210,
    isAlpha: true,
  },
  {
    id: "signal-008",
    round: 8,
    ticker: "HZN",
    title: "Influencer chatter outruns actual operating metrics",
    description:
      "Traffic and booking data both undershoot the social-media excitement around the name.",
    sector: "Travel",
    value: 160,
    isAlpha: false,
  },
];

export function getSignalForRound(round) {
  return SIGNALS.find((signal) => signal.round === round) ?? null;
}

export function toPublicSignal(signal) {
  if (!signal) {
    return null;
  }

  return {
    id: signal.id,
    round: signal.round,
    ticker: signal.ticker,
    title: signal.title,
    description: signal.description,
    sector: signal.sector,
    value: signal.value,
  };
}
