"use client";
import { useParams } from "next/navigation";
import OptimizationView from "@/components/OptimizationView";

export default function OptimizationPage() {
  const { id } = useParams<{ id: string }>();
  return <OptimizationView id={Number(id)} />;
}
