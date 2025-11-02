import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Clock, Zap, CheckCircle2, XCircle } from "lucide-react"

export default function AgentStatus() {
  // Props are globally injected
  const agents = props?.agents || [];
  const totalAgents = agents.length;
  const activeAgents = agents.filter(a => a.active).length;

  const getStatusColor = (active) => {
    return active ? "bg-green-500" : "bg-gray-400";
  }

  const formatDuration = (ms) => {
    if (!ms) return '00:00:00';
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }

  return (
    <Card className="w-full max-w-4xl my-4">
      <CardHeader className="pb-3">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Multi-Agent System Status
          </CardTitle>
          <Badge variant="outline" className="text-sm">
            {activeAgents} / {totalAgents} Active
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {agents.map((agent, index) => (
            <div 
              key={index}
              className={`flex items-center justify-between p-3 rounded-lg border transition-all ${
                agent.active 
                  ? 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800' 
                  : 'bg-gray-50 border-gray-200 dark:bg-gray-900 dark:border-gray-800'
              }`}
            >
              <div className="flex items-center gap-3 flex-1">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.active)} ${
                  agent.active ? 'animate-pulse' : ''
                }`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{agent.name}</span>
                    {agent.active ? (
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                    ) : (
                      <XCircle className="h-4 w-4 text-gray-400" />
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {agent.description || 'Agent description'}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  <span className="font-mono">{formatDuration(agent.duration || 0)}</span>
                </div>
                {agent.active && (
                  <Badge variant="secondary" className="bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">
                    Running
                  </Badge>
                )}
              </div>
            </div>
          ))}
          {totalAgents === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              No agents available
            </div>
          )}
        </div>
        
        {props.showProgress && (
          <div className="mt-4">
            <div className="flex justify-between text-xs text-muted-foreground mb-2">
              <span>System Activity</span>
              <span>{Math.round((activeAgents / totalAgents) * 100)}%</span>
            </div>
            <Progress value={(activeAgents / totalAgents) * 100} className="h-2" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}

