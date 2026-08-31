import { createFileRoute } from "@tanstack/react-router";
import { ProcessingPage } from "@/pages/ProcessingPage";

export const Route = createFileRoute("/processing")({
  component: ProcessingPage,
});
