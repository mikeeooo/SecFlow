import { ChevronRight } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { useAccounts } from "../api/hooks";
import type { AccountSummary, Decision, DecisionLabel } from "../api/types";
import { DecisionBadge } from "../components/DecisionBadge";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { Card, Chip, Field, PageHeader, Segmented, Stat, inputClass } from "../components/ui";
import { formatDateTime, formatInt } from "../lib/format";

const LAST: Record<Decision, DecisionLabel> = { Permit: "Permitted", Verify: "Needs verification", Block: "Preventive risk block" };

/** Large enough for every account in the dataset, so the whole list loads in one request. */
const ALL_ACCOUNTS = 1000;

type Sort = "name" | "activity" | "flagged";

const SORTERS: Record<Sort, (a: AccountSummary, b: AccountSummary) => number> = {
  name: (a, b) => a.display_name.localeCompare(b.display_name),
  activity: (a, b) => b.requests - a.requests || a.display_name.localeCompare(b.display_name),
  flagged: (a, b) => b.flagged - a.flagged || b.rejected - a.rejected || a.display_name.localeCompare(b.display_name),
};

export function AccountsPage() {
  const [params, setParams] = useSearchParams();
  const [draft, setDraft] = useState(params.get("q") ?? "");
  const type = params.get("account_type") ?? "";
  const sortParam = params.get("sort");
  const sort: Sort = sortParam === "activity" || sortParam === "flagged" ? sortParam : "name";
  const { data, error, isPending, refetch } = useAccounts({ q: params.get("q") ?? undefined, account_type: type || undefined, page: 1, page_size: ALL_ACCOUNTS });

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  };

  const items = useMemo(() => (data ? [...data.items].sort(SORTERS[sort]) : []), [data, sort]);

  return (
    <>
      <PageHeader title="Accounts" subtitle="Every account in the demonstration history with its activity, flags, and review outcomes. Names are generated identities." />
      {data ? (
        <Card className="mb-5 grid grid-cols-2 gap-6 p-5 md:grid-cols-4">
          <Stat label="All accounts" value={formatInt(data.all_accounts)} />
          <Stat label="Personal" value={formatInt(data.personal)} />
          <Stat label="Business" value={formatInt(data.business)} />
          <Stat label="With activity" value={formatInt(data.active)} hint="Sent at least one request" />
        </Card>
      ) : null}
      <Card className="mb-5 flex flex-wrap items-end gap-3 p-5">
        <form className="min-w-[240px] flex-1" onSubmit={(e) => { e.preventDefault(); set("q", draft.trim()); }} role="search">
          <Field label="Search accounts" htmlFor="acct-search">
            <input id="acct-search" type="search" className={inputClass} placeholder="Name or account ID" value={draft}
              onChange={(e) => { setDraft(e.target.value); if (!e.target.value) set("q", ""); }} />
          </Field>
        </form>
        <Segmented label="Account type" value={type} onChange={(v) => set("account_type", v)}
          options={[{ value: "", label: "All" }, { value: "personal", label: "Personal" }, { value: "business", label: "Business" }]} />
        <Segmented label="Sort accounts" value={sort} onChange={(v) => set("sort", v === "name" ? "" : v)}
          options={[{ value: "name", label: "Name" }, { value: "activity", label: "Most active" }, { value: "flagged", label: "Most flagged" }]} />
      </Card>
      <Card>
        {isPending ? <LoadingState rows={8} /> : error ? <ErrorState error={error} onRetry={() => void refetch()} /> : items.length === 0 ? (
          <EmptyState title="No accounts found">Search by part of a name, for example a surname, or by account ID.</EmptyState>
        ) : (
          <>
            <p className="tabular border-b border-separator px-5 py-3 text-[13px] text-label-2">
              {items.length === data.all_accounts ? `Showing all ${formatInt(items.length)} accounts` : `Showing ${formatInt(items.length)} of ${formatInt(data.all_accounts)} accounts`}
            </p>
            <ul>
              {items.map((a) => (
                <li key={a.account_id} className="border-b border-separator last:border-0">
                  <Link to={`/accounts/${a.account_id}`} className="grid grid-cols-[1fr_auto] items-center gap-3 px-5 py-3 hover:bg-surface-2 sm:grid-cols-[minmax(0,1fr)_150px_150px_auto]">
                    <div className="min-w-0">
                      <p className="truncate text-[15px] font-medium">{a.display_name}</p>
                      <p className="text-[12px] text-label-2">{a.account_id} · {a.account_type} · {a.home_country}</p>
                    </div>
                    <div className="tabular hidden text-[13px] text-label-2 sm:block">
                      {formatInt(a.requests)} requests<br />{formatInt(a.flagged)} flagged
                      {a.rejected ? <> · <span className="text-block">{formatInt(a.rejected)} rejected</span></> : null}
                    </div>
                    <div className="hidden text-[12px] text-label-2 sm:block">{a.last_activity ? formatDateTime(a.last_activity) : "No activity"}</div>
                    <div className="flex items-center gap-1">
                      {a.rejected ? <span className="sm:hidden"><Chip tone="block">{a.rejected} rejected</Chip></span> : null}
                      {a.last_decision ? <DecisionBadge size="sm" label={LAST[a.last_decision]} /> : null}
                      <ChevronRight aria-hidden className="size-4 text-label-2" />
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          </>
        )}
      </Card>
    </>
  );
}
