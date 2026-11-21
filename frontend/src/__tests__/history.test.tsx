import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AccountSummary, ReviewHistoryItem } from "../api/types";
import { AccountsPage } from "../pages/Accounts";
import { ReviewHistoryPage } from "../pages/ReviewHistory";
import { mockFetch, renderAt } from "./helpers";

afterEach(() => vi.unstubAllGlobals());

const review = (id: string, action: "approve" | "reject", overrides: Partial<ReviewHistoryItem> = {}): ReviewHistoryItem => ({
  review_id: id, request_id: `SIM-${id}`, action, reason: "Beneficiary could not be verified", actor: "Ana",
  created_at: "2026-11-21T10:00:00.00Z", original_decision: "Verify", original_decision_label: "Needs verification",
  original_risk_score: 72.4, resulting_outcome: action === "reject" ? "rejected_after_review" : "approved_after_review",
  origin: "simulation", event_time: "2026-11-21T09:59:00Z", sender_id: "ACC-0001", sender_name: "Amara Okafor",
  recipient_id: "CP-00001", recipient_name: "Harbor Lane Utilities", amount: 6000, currency: "EUR",
  investigation_available: true, ...overrides,
});

const account = (i: number): AccountSummary => ({
  account_id: `ACC-${String(i).padStart(4, "0")}`, display_name: `Person ${String(i).padStart(3, "0")}`,
  account_type: i % 2 ? "personal" : "business", home_country: "DE", created_at: "2026-01-01T00:00:00Z",
  requests: i, flagged: i % 3, rejected: i === 7 ? 2 : 0, last_activity: null, last_decision: null,
});

describe("Review history", () => {
  it("lists saved rejections and filters by outcome", async () => {
    const calls = mockFetch((url) => {
      const rejectOnly = url.includes("action=reject");
      const items = rejectOnly ? [review("REV-00002", "reject")] : [review("REV-00002", "reject"), review("REV-00001", "approve")];
      return { body: { items, total: items.length, page: 1, page_size: 50, approved: 1, rejected: 1, persisted: true, storage_note: null } };
    });
    renderAt(<ReviewHistoryPage />, "/history");
    expect(await screen.findByText("Rejected after review")).toBeInTheDocument();
    expect(screen.getByText("Approved after review")).toBeInTheDocument();
    expect(screen.getByText("Saved")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: "Rejected" }));
    await waitFor(() => expect(screen.queryByText("Approved after review")).not.toBeInTheDocument());
    expect(calls.some((c) => c.url.includes("action=reject"))).toBe(true);
  });

  it("explains how to create the first review", async () => {
    mockFetch(() => ({ body: { items: [], total: 0, page: 1, page_size: 50, approved: 0, rejected: 0, persisted: true, storage_note: null } }));
    renderAt(<ReviewHistoryPage />, "/history");
    expect(await screen.findByText("No reviews yet")).toBeInTheDocument();
  });
});

describe("Accounts", () => {
  it("loads and shows every account in one list", async () => {
    const items = Array.from({ length: 400 }, (_, i) => account(i + 1));
    const calls = mockFetch(() => ({ body: { items, total: 400, page: 1, page_size: 1000, all_accounts: 400, personal: 200, business: 200, active: 400 } }));
    renderAt(<AccountsPage />, "/accounts");
    expect(await screen.findByText("Showing all 400 accounts")).toBeInTheDocument();
    expect(screen.getAllByRole("link")).toHaveLength(400);
    expect(calls[0].url).toContain("page_size=1000");
    expect(screen.queryByRole("navigation", { name: "Pagination" })).not.toBeInTheDocument();
  });
});
