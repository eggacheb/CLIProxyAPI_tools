// Package mcp provides MCP XML Bridge functionality.
// This package handles the conversion of MCP tool calls (prefixed with "mcp__")
// between XML format and standard tool_use format, enabling MCP tools to work
// through text-based channels that don't support native function calling.
package mcp

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync/atomic"
)

// MCP_XML_ENV is the environment variable name to control MCP XML mode.
// Set to "0", "false", "no", or "off" to disable. Enabled by default.
const MCP_XML_ENV = "AG2API_MCP_XML_ENABLED"

var mcpToolUseIDCounter uint64

// IsMcpXmlEnabled checks if MCP XML mode is enabled.
// Returns true by default. Set AG2API_MCP_XML_ENABLED=false to disable.
func IsMcpXmlEnabled() bool {
	raw := os.Getenv(MCP_XML_ENV)
	if raw == "" {
		return true // 默认启用
	}
	v := strings.ToLower(strings.TrimSpace(raw))
	return v != "0" && v != "false" && v != "no" && v != "off"
}

// IsMcpToolName checks if a tool name is an MCP tool (prefixed with "mcp__").
func IsMcpToolName(name string) bool {
	return strings.HasPrefix(name, "mcp__")
}

// GetMcpToolNames extracts MCP tool names from a list of tools (gjson results).
func GetMcpToolNames(tools []string) []string {
	var result []string
	for _, name := range tools {
		if IsMcpToolName(name) {
			result = append(result, name)
		}
	}
	return result
}

// BuildMcpXmlSystemPrompt generates a system prompt instructing the model
// to use XML format for MCP tool calls instead of normal function calling.
func BuildMcpXmlSystemPrompt(mcpTools []McpTool) string {
	if len(mcpTools) == 0 {
		return ""
	}

	var lines []string
	lines = append(lines, "==== MCP XML 工具调用（仅 mcp__*） ====")
	lines = append(lines, "当你需要调用名称以 `mcp__` 开头的 MCP 工具时：")
	lines = append(lines, "1) 不要使用 tool_use/function_call（因为该链路会报错）。")
	lines = append(lines, "2) 直接输出一个 XML 块（只输出 XML，不要解释/不要 markdown）。")
	lines = append(lines, "3) XML 的根标签必须是工具名，内容必须是 JSON（对象/数组），表示该工具的入参。")
	lines = append(lines, "")
	lines = append(lines, "示例：")
	lines = append(lines, `<mcp__server__tool>{"arg":"value"}</mcp__server__tool>`)
	lines = append(lines, "")
	lines = append(lines, "工具执行完成后，我会把结果以如下 XML 返回给你：")
	lines = append(lines, `<mcp_tool_result>{"name":"mcp__server__tool","tool_use_id":"toolu_xxx","result":"...","is_error":false}</mcp_tool_result>`)
	lines = append(lines, "")
	lines = append(lines, "当 is_error 为 true 时，表示该工具执行失败，result 内容为错误信息。")
	lines = append(lines, "")
	lines = append(lines, "对于非 `mcp__*` 工具：继续使用正常的工具调用机制。")
	lines = append(lines, "")
	lines = append(lines, "可用 MCP 工具列表（name / description / input_schema）：")

	for _, tool := range mcpTools {
		if !IsMcpToolName(tool.Name) {
			continue
		}
		desc := tool.Description
		line := fmt.Sprintf("- %s", tool.Name)
		if desc != "" {
			line += ": " + desc
		}
		lines = append(lines, line)
		if tool.InputSchema != "" {
			lines = append(lines, "  input_schema: "+tool.InputSchema)
		}
	}

	return strings.Join(lines, "\n")
}

// McpTool represents an MCP tool definition.
type McpTool struct {
	Name        string
	Description string
	InputSchema string // JSON string of input schema
}

// McpToolCall represents a parsed MCP tool call from XML.
type McpToolCall struct {
	Name  string
	Input map[string]interface{}
	ID    string // Generated tool_use ID
}

// BuildMcpToolResultXml creates an XML representation of a tool result.
func BuildMcpToolResultXml(toolName, toolUseID, result string, isError bool) string {
	payload := map[string]interface{}{
		"name":        toolName,
		"tool_use_id": toolUseID,
		"result":      result,
		"is_error":    isError,
	}
	jsonBytes, _ := json.Marshal(payload)
	return fmt.Sprintf("<mcp_tool_result>%s</mcp_tool_result>", string(jsonBytes))
}

// MakeToolUseID generates a unique tool use ID.
func MakeToolUseID(name string) string {
	return fmt.Sprintf("%s-%d", name, atomic.AddUint64(&mcpToolUseIDCounter, 1))
}

// XmlStreamParser parses streaming text for MCP XML tool calls.
type XmlStreamParser struct {
	toolNames map[string]bool
	buffer    string
}

// NewXmlStreamParser creates a new parser for the given MCP tool names.
func NewXmlStreamParser(toolNames []string) *XmlStreamParser {
	names := make(map[string]bool)
	for _, name := range toolNames {
		if name != "" {
			names[name] = true
		}
	}
	return &XmlStreamParser{
		toolNames: names,
		buffer:    "",
	}
}

// ParseResult represents a parsed chunk from the stream.
type ParseResult struct {
	Type  string                 // "text" or "tool"
	Text  string                 // For type "text"
	Name  string                 // For type "tool"
	Input map[string]interface{} // For type "tool"
}

// PushText adds text to the parser and returns any complete results.
func (p *XmlStreamParser) PushText(text string) []ParseResult {
	var results []ParseResult
	if text == "" {
		return results
	}
	p.buffer += text

	for {
		index, name := p.findNextToolStartIndex()
		if index == -1 || name == "" {
			// Check for partial tag at end
			emit, keep := p.splitBufferForPartialTag()
			if emit != "" {
				results = append(results, ParseResult{Type: "text", Text: emit})
			}
			p.buffer = keep
			break
		}

		// Emit text before the tool tag
		if index > 0 {
			results = append(results, ParseResult{Type: "text", Text: p.buffer[:index]})
			p.buffer = p.buffer[index:]
		}

		// Try to find complete tool call
		closeEnd := p.findCloseTagEndIndex(name)
		if closeEnd == -1 {
			break // Incomplete, wait for more data
		}

		xml := p.buffer[:closeEnd]
		p.buffer = p.buffer[closeEnd:]

		// Parse the XML
		parsed, ok := TryParseMcpToolCallXml(xml, name)
		if ok {
			results = append(results, ParseResult{
				Type:  "tool",
				Name:  parsed.Name,
				Input: parsed.Input,
			})
		} else {
			results = append(results, ParseResult{Type: "text", Text: xml})
		}
	}

	return results
}

// Flush returns any remaining buffered content.
func (p *XmlStreamParser) Flush() []ParseResult {
	var results []ParseResult
	if p.buffer != "" {
		results = append(results, ParseResult{Type: "text", Text: p.buffer})
		p.buffer = ""
	}
	return results
}

func (p *XmlStreamParser) findNextToolStartIndex() (int, string) {
	best := -1
	bestName := ""
	for name := range p.toolNames {
		open := "<" + name
		idx := strings.Index(p.buffer, open)
		if idx == -1 {
			continue
		}
		// Check boundary character
		if idx+len(open) < len(p.buffer) {
			ch := p.buffer[idx+len(open)]
			if ch != '>' && ch != '/' && ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r' {
				continue
			}
		}
		if best == -1 || idx < best {
			best = idx
			bestName = name
		}
	}
	return best, bestName
}

func (p *XmlStreamParser) findCloseTagEndIndex(name string) int {
	needle := "</" + name
	idx := strings.Index(p.buffer, needle)
	if idx == -1 {
		return -1
	}
	after := idx + len(needle)
	if after >= len(p.buffer) {
		return -1 // Incomplete
	}
	// Find the closing >
	gtIdx := strings.Index(p.buffer[after:], ">")
	if gtIdx == -1 {
		return -1
	}
	return after + gtIdx + 1
}

func (p *XmlStreamParser) splitBufferForPartialTag() (string, string) {
	lastLt := strings.LastIndex(p.buffer, "<")
	if lastLt == -1 {
		return p.buffer, ""
	}
	tail := p.buffer[lastLt:]
	if p.isPossibleToolTagPrefix(tail) {
		return p.buffer[:lastLt], tail
	}
	return p.buffer, ""
}

func (p *XmlStreamParser) isPossibleToolTagPrefix(text string) bool {
	if !strings.HasPrefix(text, "<") {
		return false
	}
	for name := range p.toolNames {
		open := "<" + name
		if strings.HasPrefix(open, text) {
			return true
		}
		close := "</" + name
		if strings.HasPrefix(close, text) {
			return true
		}
	}
	return false
}

// TryParseMcpToolCallXml attempts to parse an XML tool call.
func TryParseMcpToolCallXml(xmlText, toolName string) (*McpToolCall, bool) {
	if toolName == "" || xmlText == "" {
		return nil, false
	}

	// Match opening tag
	openPattern := regexp.MustCompile(`^\s*<` + regexp.QuoteMeta(toolName) + `(\s[^>]*)?>`)
	closePattern := regexp.MustCompile(`</` + regexp.QuoteMeta(toolName) + `\s*>\s*$`)

	openMatch := openPattern.FindStringIndex(xmlText)
	closeMatch := closePattern.FindStringIndex(xmlText)

	if openMatch == nil || closeMatch == nil {
		return nil, false
	}

	// Extract inner content
	inner := xmlText[openMatch[1]:closeMatch[0]]
	inner = strings.TrimSpace(inner)

	if inner == "" {
		return &McpToolCall{
			Name:  toolName,
			Input: make(map[string]interface{}),
			ID:    MakeToolUseID(toolName),
		}, true
	}

	// Try to parse as JSON
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(inner), &parsed); err != nil {
		return nil, false
	}

	return &McpToolCall{
		Name:  toolName,
		Input: parsed,
		ID:    MakeToolUseID(toolName),
	}, true
}
