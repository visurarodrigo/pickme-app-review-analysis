## 📈 Mini Report: Key Insights

### 1. Overall Sentiment
- Out of **1,440 analyzed English reviews**, **58.8% are Negative (847)** and **41.2% are Positive (593)**.
- Negative reviews dominate - this is a common pattern in app-store data, where unhappy users are more motivated to write.

### 2. Trend Over Time
- Negative sentiment **outnumbers positive in every single week** of the year (Aug 2025 – Aug 2026).
- The **largest negative spike (50+ reviews in one week) appears in late March 2026**, driven by a wave of complaints about **event ticket refunds (NeYo concert), pricing, and trust**.
- Smaller negative spikes in **Sep 2025** and **Nov–Dec 2025** coincide with complaint clusters about **food delivery delays** and the newly introduced **driver bidding feature**.
- Positive reviews stay **flat and stable (5–20 per week)** - loyal happy users exist, but they cannot offset the complaint waves.

### 3. Main Pain Points (Negative Reviews)
- **Food delivery reliability** - orders waiting 1-4 hours with no rider assigned, no in-app cancellation.
- **Pricing transparency** - surge charges, driver bidding (+100 to +1000), final fare higher than the estimate, hidden fees.
- **Driver issues** - late arrivals, st-minute cancellations, overcharging, refusing card payments, taking longer routes.
- **Customer support & refunds** - unreachable hotline, unresolved refunds, complaints closed without resolution.
- **Privacy & safety** - phone numbers visible to drivers, harassment after rides.
- **App bugs** - GPS/location inaccuracy, login failures, crashes on older devices.

### 4. Main Strengths (Positive Reviews)
- **Convenience** - easy booking and island-wide coverage.
- **Driver kindness** - many reviews praise polite, honest, and helpful drivers.
- **Affordability** - fair and reasonable pricing for many users.
- **Reliability in emergencies** - quick vehicle access when urgently needed.

### 5. Language Coverage
- The dataset is multilingual: **1,440 English, 40 Sinhala, 4 Tamil**.
- Non-English reviews are kept in the dataset for transparency but excluded from sentiment scoring, since the pretrained model is English-focused.

### 6. Business Takeaway
> PickMe's core ride-hailing experience is appreciated, but **food delivery reliability, pricing transparency, refunds, and customer support** are the biggest drivers of negative sentiment - and complaint waves (like the March 2026 refund scandal) create sharp, visible spikes in the weekly trend.

Dashboard Preview & Final Dataset
- If you need to see visuals, you can use this [dashboard preview folder](./dashboard-preview).
- If you need to check the dataset, you can see the [final dataset](./data/pickme_reviews_with_sentiment.csv).
