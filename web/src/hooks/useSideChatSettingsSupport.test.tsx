import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { authenticatedFetch } from "@/lib/identity";
import { useSideChatSettingsSupport } from "./useSideChatSettingsSupport";

vi.mock("@/lib/identity", () => ({ authenticatedFetch: vi.fn() }));
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it.each([
  [404, {}, false],
  [200, { settings: false }, false],
  [200, { settings: true }, true],
] as const)("negotiates server and runner support (%s, %j)", async (status, body, expected) => {
  vi.mocked(authenticatedFetch).mockResolvedValue(new Response(JSON.stringify(body), { status }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const { result } = renderHook(() => useSideChatSettingsSupport("parent", true), { wrapper });
  await waitFor(() =>
    expect(client.getQueryData(["side-chat-settings-support", "parent"])).toBe(expected),
  );
  expect(result.current).toBe(expected);
  expect(authenticatedFetch).toHaveBeenCalledWith("/v1/sessions/parent/side_chat_capabilities");
  client.clear();
});
