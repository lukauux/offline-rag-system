"use client"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
} from "@/components/ui/dropdown-menu"
import { Settings2, Sparkles, Target, Layers } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface SearchSettingsProps {
  searchType: "similarity" | "mmr" | "similarity_score"
  onSearchTypeChange: (type: "similarity" | "mmr" | "similarity_score") => void
  useMultiQuery: boolean
  onUseMultiQueryChange: (value: boolean) => void
  topK: number
  onTopKChange: (value: number) => void
}

export function SearchSettings({
  searchType,
  onSearchTypeChange,
  useMultiQuery,
  onUseMultiQueryChange,
  topK,
  onTopKChange,
}: SearchSettingsProps) {
  const getSearchTypeLabel = () => {
    switch (searchType) {
      case "mmr":
        return "MMR (Diverse)"
      case "similarity":
        return "Similarity"
      case "similarity_score":
        return "Scored"
      default:
        return "Search"
    }
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2 bg-transparent">
          <Settings2 className="h-4 w-4" />
          {getSearchTypeLabel()}
          <Badge variant="secondary" className="ml-1">
            {topK}
          </Badge>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        <DropdownMenuLabel>Search Strategy</DropdownMenuLabel>
        <DropdownMenuSeparator />

        <DropdownMenuRadioGroup value={searchType} onValueChange={(v) => onSearchTypeChange(v as any)}>
          <DropdownMenuRadioItem value="mmr" className="gap-2">
            <Sparkles className="h-4 w-4" />
            <div className="flex-1">
              <div className="font-medium">MMR (Recommended)</div>
              <div className="text-xs text-muted-foreground">Diverse, relevant results</div>
            </div>
          </DropdownMenuRadioItem>

          <DropdownMenuRadioItem value="similarity" className="gap-2">
            <Target className="h-4 w-4" />
            <div className="flex-1">
              <div className="font-medium">Similarity</div>
              <div className="text-xs text-muted-foreground">Most similar results</div>
            </div>
          </DropdownMenuRadioItem>

          <DropdownMenuRadioItem value="similarity_score" className="gap-2">
            <Layers className="h-4 w-4" />
            <div className="flex-1">
              <div className="font-medium">Scored Similarity</div>
              <div className="text-xs text-muted-foreground">With relevance scores</div>
            </div>
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>

        <DropdownMenuSeparator />

        <DropdownMenuCheckboxItem checked={useMultiQuery} onCheckedChange={onUseMultiQueryChange}>
          <div className="flex-1">
            <div className="font-medium">Multi-Query Mode</div>
            <div className="text-xs text-muted-foreground">Search with query variations</div>
          </div>
        </DropdownMenuCheckboxItem>

        <DropdownMenuSeparator />
        <DropdownMenuLabel>Results Count</DropdownMenuLabel>

        <DropdownMenuRadioGroup value={topK.toString()} onValueChange={(v) => onTopKChange(Number.parseInt(v))}>
          <DropdownMenuRadioItem value="3">3 results</DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="5">5 results</DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="10">10 results</DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
