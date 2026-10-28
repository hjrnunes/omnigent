import { useQuery } from "@tanstack/react-query";
import { authenticatedFetch } from "@/lib/identity";

export function useSideChatSettingsSupport(sessionId: string | null, native: boolean): boolean {
  const { data } = useQuery({
    queryKey: ["side-chat-settings-support", sessionId],
    enabled: native && sessionId !== null,
    staleTime: 30_000,
    refetchInterval: (query) => (query.state.data === true ? false : 10_000),
    queryFn: async () => {
      try {
        const response = await authenticatedFetch(
          `/v1/sessions/${encodeURIComponent(sessionId!)}/side_chat_capabilities`,
        );
        if (!response.ok) return false;
        return (await response.json()).settings === true;
      } catch {
        return false;
      }
    },
  });
  return !native || data === true;
}
