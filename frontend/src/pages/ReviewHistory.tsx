import { CheckCircle2, XCircle } from "lucide-react";
import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import { useReviewHistory } from "../api/hooks";
import type { ReviewHistoryItem } from "../api/types";
import { DecisionBadge } from "../components/DecisionBadge";
import { Pagination } from "../components/Pagination";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { Card, Chip, Field, PageHeader, Segmented, Stat, inputClass } from "../components/ui";
import { OUTCOME_TEXT, formatDateTime, formatInt, formatMoney, formatScore } from "../lib/format";

const PAGE_SIZE = 50;

export function ReviewHistoryPage() {
  const [params, setParams] = useSearchParams();
  const [draft, setDraft] = useState(params.get("q") ?? "");
  const action = params.get("action") ?? "";
  const sort = params.get("sort") === "oldest" ? "oldest" : "newest";
  const page = Number(params.get("page") ?? "1") || 1;
  const { data, error, isPending, refetch } = useReviewHistory({
    action: action || undefined, q: params.get("q") ?? undefined, sort, page, page_size: PAGE_SIZE,
  });

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== "page") next.delete("page");
    setParams(next, { replace: key !== "page" });
  };

  return (
    <>
      <PageHeader
        title="Review history"
        subtitle="Every analyst decision on a held payment. Rejected and approved reviews are saved to a local history file and stay here after a restart."
      />
      {data ? (
        <Card className="mb-5 grid grid-cols-2 gap-6 p-5 md:grid-cols-4">
          <Stat label="All reviews" value={formatInt(data.approved + data.rejected)} />
          <Stat label="Rejected" value={formatInt(data.rejected)} tone="block" />
          <Stat label="Approved" value={formatInt(data.approved)} tone="permit" />
          <Stat label="Storage" value={data.persisted ? "Saved" : "Memory"} hint={data.persisted ? "Survives a restart" : data.storage_note ?? "Not persisted"} tone={data.persisted ? undefined : "verify"} />
        </Card>
      ) : null}
      <Card className="mb-5 flex flex-wrap items-end gap-3 p-5">
        <form className="min-w-[240px] flex-1" onSubmit={(e) => { e.preventDefault(); set("q", draft.trim()); }} role="search">
          <Field label="Search history" htmlFor="history-search">
            <input id="history-search" type="search" className={inputClass} placeholder="Name, account, request ID, reviewer or reason" value={draft}
              onChange={(e) => { setDraft(e.target.value); if (!e.target.value) set("q", ""); }} />
          </Field>
        </form>
        <Segmented label="Review outcome" value={action} onChange={(v) => set("action", v)}
          options={[{ value: "", label: "All" }, { value: "reject", label: "Rejected" }, { value: "approve", label: "Approved" }]} />
        <Segmented label="Order" value={sort} onChange={(v) => set("sort", v === "newest" ? "" : v)}
          options={[{ value: "newest", label: "Newest" }, { value: "oldest", label: "Oldest" }]} />
      </Card>
      <Card>
        {isPending ? <LoadingState rows={6} label="Loading review history" /> : error ? <ErrorState error={error} onRetry={() => void refetch()} /> : data.items.length === 0 ? (
          <EmptyState title={action || params.get("q") ? "No reviews match" : "No reviews yet"}>
            Open a payment marked Needs verification from <Link className="text-accent" to="/transactions?decision=Verify&review_status=pending">Transactions</Link> and approve or reject it. The decision appears here.
          </EmptyState>
        ) : (
          <>
            <ul>{data.items.map((item) => <HistoryRow key={item.review_id} item={item} />)}</ul>
            <Pagination page={data.page} pageSize={data.page_size} total={data.total} onPage={(p) => set("page", String(p))} />
          </>
        )}
      </Card>
    </>
  );
}

function HistoryRow({ item }: { item: ReviewHistoryItem }) {
  const rejected = item.action === "reject";
  const title = (
    <span className="flex flex-wrap items-center gap-2">
      {rejected ? <XCircle aria-hidden className="size-4 text-block" /> : <CheckCircle2 aria-hidden className="size-4 text-permit" />}
      <span className="font-semibold">{OUTCOME_TEXT[item.resulting_outcome]}</span>
      <span className="text-label-2">·</span>
      <span className="min-w-0 truncate">{item.sender_name} → {item.recipient_name}</span>
    </span>
  );
  return (
    <li className="border-b border-separator px-5 py-4 last:border-0">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <p className="text-[15px]">
            {item.investigation_available ? <Link to={`/investigations/${encodeURIComponent(item.request_id)}`} className="hover:underline">{title}</Link> : title}
          </p>
          <p className="text-[14px]">“{item.reason}”</p>
          <p className="text-[12px] text-label-2">
            {item.actor} · reviewed {formatDateTime(item.created_at)} · {item.review_id} · request {item.request_id} from {formatDateTime(item.event_time)}
            {item.investigation_available ? null : " · the simulated request ended with an earlier server run"}
          </p>
        </div>
        <div className="flex flex-col items-end gap-1.5 text-right">
          <span className="tabular text-[15px] font-semibold">{formatMoney(item.amount)}</span>
          <span className="tabular text-[12px] text-label-2">risk {formatScore(item.original_risk_score)} / 100</span>
          <div className="flex flex-wrap justify-end gap-1">
            <DecisionBadge size="sm" label={item.original_decision_label} />
            <Chip tone={item.origin === "simulation" ? "accent" : "neutral"}>{item.origin === "simulation" ? "Simulated" : "History"}</Chip>
          </div>
        </div>
      </div>
    </li>
  );
}
