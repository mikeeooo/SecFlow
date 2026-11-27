import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SimulatorPage } from "../pages/Simulator";
import { mockFetch, renderAt, summary } from "./helpers";

afterEach(() => vi.unstubAllGlobals());

const options = {
  simulation_time: "2026-04-15T10:00:00Z", currencies: ["MDL"], auth_methods: ["biometric"], countries: ["MD", "NL"],
  payment_schemes: ["p2p_alias", "p2p_iban", "p2b_qr_static", "p2b_qr_dynamic", "request_to_pay"],
  limits: { personal: { per_transaction: 5000, per_day: 10000, per_month: 30000 }, business: { per_transaction: 100000, per_day: 1000000, per_month: null } },
  flagged_beneficiaries: [],
  sender: { account_id: "ACC-0001", account_type: "personal", max_amount: 5000, home_country: "MD", known_devices: ["DEV-0001-A"], recent_recipients: [{ id: "CP-00001", name: "Valea Verde Utilities", kind: "counterparty" }], median_amount: 400 },
};
const detail = {
  ...summary("SIM-x", 11), explanation: "Permitted: risk 31.2 is below the verification threshold 61.", status_notes: [],
  component_details: [], reasons: [], evidence: [], thresholds: { verify: 61, block: 97, min_corroborating_families: 2, source: "validated" },
  model_version: "m", policy_version: "p", operating_point_version: "o", assessed_at: "2026-10-28T10:00:00Z",
  device_id: "DEV-0001-A", session_id: "S", auth_method: "biometric", ip_country: "MD", channel: "mobile_app",
  request_to_pay_id: null, qr_payload_merchant_id: null, recipient_alias: null, facts: {}, idempotent_replay: false,
};

function baseHandler(simulate: (body: Record<string, unknown>) => { status?: number; body?: unknown }) {
  return mockFetch((url, init) => {
    if (url.startsWith("/api/parties")) return { body: [{ id: "ACC-0001", name: "Ion Rusu", kind: "account", detail: "personal" }] };
    if (url.startsWith("/api/simulator/options")) return { body: options };
    if (url === "/api/simulations") return simulate(JSON.parse(String(init?.body)));
    return { body: {} };
  });
}

describe("Payment Simulator", () => {
  it("does not submit anything on load (a refresh cannot resubmit)", async () => {
    const calls = baseHandler(() => ({ status: 201, body: detail }));
    renderAt(<SimulatorPage />);
    await screen.findByDisplayValue("Ion Rusu");
    expect(calls.filter((c) => c.url === "/api/simulations")).toHaveLength(0);
  });

  it("validates required fields before sending", async () => {
    const calls = baseHandler(() => ({ status: 201, body: detail }));
    renderAt(<SimulatorPage />);
    await screen.findByDisplayValue("Ion Rusu");
    await userEvent.click(screen.getByRole("button", { name: "Submit for assessment" }));
    expect(await screen.findByText("Enter an amount greater than zero.")).toBeInTheDocument();
    expect(screen.getByText(/Choose a recipient/)).toBeInTheDocument();
    expect(calls.filter((c) => c.url === "/api/simulations")).toHaveLength(0);
  });

  it("retries a failed submission with the same request ID, then uses a new ID", async () => {
    const ids: string[] = [];
    let attempt = 0;
    baseHandler((body) => {
      ids.push(String(body.request_id));
      attempt += 1;
      return attempt === 1 ? { status: 0 } : { status: 201, body: detail };
    });
    renderAt(<SimulatorPage />);
    await screen.findByDisplayValue("Ion Rusu");
    await userEvent.click(await screen.findByRole("button", { name: /Ordinary payment/ }));
    await userEvent.click(screen.getByRole("button", { name: "Submit for assessment" }));
    await userEvent.click(await screen.findByRole("button", { name: /Retry same request/ }));
    expect(await screen.findByText(/below the verification threshold/)).toBeInTheDocument();
    expect(ids).toHaveLength(2);
    expect(ids[0]).toBe(ids[1]);
    await waitFor(() => expect(screen.getByText(/^Request ID /).textContent).not.toContain(ids[0]));
    expect(screen.getByRole("link", { name: /Open investigation/ })).toHaveAttribute("href", "/investigations/SIM-x");
  });
});

describe("Payment Simulator · MIA Plăți Instant", () => {
  it("sends MDL with the chosen payment scheme", async () => {
    const bodies: Record<string, unknown>[] = [];
    baseHandler((body) => { bodies.push(body); return { status: 201, body: detail }; });
    renderAt(<SimulatorPage />);
    await screen.findByDisplayValue("Ion Rusu");
    await userEvent.click(await screen.findByRole("button", { name: /Ordinary payment/ }));
    await userEvent.selectOptions(screen.getByLabelText("MIA payment scheme"), "request_to_pay");
    await userEvent.click(screen.getByRole("button", { name: "Submit for assessment" }));
    await screen.findByText(/below the verification threshold/);
    expect(bodies[0]).toMatchObject({ currency: "MDL", payment_scheme: "request_to_pay", qr_payload_merchant_id: null });
    expect(screen.getByText(/Payment link ·/)).toBeInTheDocument(); // the scheme the server recorded
  });

  it("refuses a personal payment above the 5,000 MDL limit before sending", async () => {
    const calls = baseHandler(() => ({ status: 201, body: detail }));
    renderAt(<SimulatorPage />);
    await screen.findByDisplayValue("Ion Rusu");
    await userEvent.click(await screen.findByRole("button", { name: /Ordinary payment/ }));
    const amount = screen.getByLabelText("Amount (MDL)");
    await userEvent.clear(amount);
    await userEvent.type(amount, "5000.01");
    await userEvent.click(screen.getByRole("button", { name: "Submit for assessment" }));
    expect(await screen.findByText(/at most MDL\s5,000\.00 per payment/)).toBeInTheDocument();
    expect(calls.filter((c) => c.url === "/api/simulations")).toHaveLength(0);
  });

  it("asks for the QR merchant only for QR schemes", async () => {
    baseHandler(() => ({ status: 201, body: detail }));
    renderAt(<SimulatorPage />);
    await screen.findByDisplayValue("Ion Rusu");
    expect(screen.queryByLabelText(/Merchant named in the QR code/)).not.toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("MIA payment scheme"), "p2b_qr_static");
    expect(screen.getByLabelText(/Merchant named in the QR code/)).toBeInTheDocument();
  });
});
