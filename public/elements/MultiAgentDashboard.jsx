import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Clock, Activity, Zap, Users, BarChart3 } from "lucide-react"

export default function MultiAgentDashboard() {
  // Props are globally injected (not function argument)
  const initialAgents = props?.agents || [];
  const stats = props?.stats || {};
  
  // State to track real-time durations
  const [agents, setAgents] = useState(initialAgents);
  const [startTime] = useState(Date.now());
  
  // Update agents with real-time durations
  useEffect(() => {
    const interval = setInterval(() => {
      setAgents(currentAgents => {
        return currentAgents.map(agent => {
          if (agent.active && agent.start_time) {
            // Calculate elapsed time if agent is active
            // start_time is in seconds (from Python), convert to ms
            const startTimeMs = agent.start_time * 1000;
            const now = Date.now();
            // Calculate base duration + elapsed time
            const baseDuration = agent.base_duration || agent.duration || 0;
            const elapsed = Math.floor(now - startTimeMs);
            return {
              ...agent,
              duration: baseDuration + elapsed
            };
          }
          // For inactive agents, keep stored duration
          return agent;
        });
      });
    }, 1000); // Update every second
    
    return () => clearInterval(interval);
  }, []);
  
  // Update when props change
  useEffect(() => {
    const newAgents = props?.agents || [];
    // Debug: log props
    if (newAgents.length > 0) {
      console.log('MultiAgentDashboard: Received agents:', newAgents);
      console.log('Active agents:', newAgents.filter(a => a.active));
    }
    setAgents(newAgents);
  }, [props?.agents]);
  
  const activeAgents = agents.filter(a => a.active).length;
  const totalAgents = agents.length;
  const totalDuration = agents.reduce((sum, a) => sum + (a.duration || 0), 0);

  const getAgentStatus = (agent) => {
    if (agent.active) {
      return {
        color: "text-green-600",
        bg: "bg-green-50",
        border: "border-green-200",
        icon: Activity,
        label: "Active"
      };
    }
    return {
      color: "text-gray-500",
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: Clock,
      label: "Inactive"
    };
  };

  const formatDuration = (ms) => {
    if (!ms || ms === 0) return '00:00:00';
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4 my-4">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Agents</p>
                <p className="text-2xl font-bold">{activeAgents}/{totalAgents}</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                <Zap className="h-6 w-6 text-primary" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Runtime</p>
                <p className="text-xl font-bold font-mono">{formatDuration(totalDuration)}</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-blue-500/10 flex items-center justify-center">
                <Clock className="h-6 w-6 text-blue-500" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Messages</p>
                <p className="text-2xl font-bold">{stats.totalMessages || 0}</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-purple-500/10 flex items-center justify-center">
                <Users className="h-6 w-6 text-purple-500" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                <p className="text-2xl font-bold">{stats.successRate || 100}%</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-green-500/10 flex items-center justify-center">
                <BarChart3 className="h-6 w-6 text-green-500" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Agent List */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Agent Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {agents.map((agent, index) => {
              const status = getAgentStatus(agent);
              const StatusIcon = status.icon;
              
              return (
                <div
                  key={index}
                  className={`flex items-center justify-between p-4 rounded-lg border transition-all hover:shadow-md ${
                    agent.active 
                      ? `${status.bg} ${status.border}` 
                      : `${status.bg} ${status.border} opacity-60`
                  }`}
                >
                  <div className="flex items-center gap-4 flex-1">
                    <div className={`h-10 w-10 rounded-lg ${status.bg} flex items-center justify-center ${
                      agent.active ? 'animate-pulse' : ''
                    }`}>
                      <StatusIcon className={`h-5 w-5 ${status.color}`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{agent.name}</span>
                        <Badge 
                          variant={agent.active ? "default" : "secondary"}
                          className={agent.active ? "bg-green-500" : ""}
                        >
                          {status.label}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {agent.description || 'No description available'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className="text-xs text-muted-foreground mb-1">Runtime</div>
                      <div className="font-mono text-sm font-medium">
                        {formatDuration(agent.duration || 0)}
                      </div>
                    </div>
                    {agent.active && (
                      <div className="h-2 w-2 rounded-full bg-green-500 animate-ping" />
                    )}
                  </div>
                </div>
              );
            })}
            {totalAgents === 0 && (
              <div className="text-center py-12 text-muted-foreground">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No agents configured</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

