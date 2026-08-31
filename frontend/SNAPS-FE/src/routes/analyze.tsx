import { createFileRoute } from "@tanstack/react-router";
import { UploadPage } from "@/pages/UploadPage";

export const Route = createFileRoute("/analyze")({
  component: UploadPage,
});
