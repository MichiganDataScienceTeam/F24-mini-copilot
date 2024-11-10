# p-transform

Promised out of order transform.

## Usage

Builds a out-of-order [Duplex](https://nodejs.org/api/stream.html#class-streamduplex) using a [p-queue](https://github.com/sindresorhus/p-queue) parallel queue.
`transform` implementation must be sync or return a promise. Callback is not supported.

Promisified `pipeline` and `transform` shortcut are provided for convenience.

```
import { OutOfOrder, transform, pipeline, passthrough, filter } from 'p-transform';

await pipeline(
  new OutOfOrder(
    { transform: async file => file },
    { concurrency: 7 },
  ).duplex(() => console.log('done')),
  passthrough(async file => {}, () => console.log('done')),
  filter(async file => true, () => console.log('done')),
  transform(async file => file, () => console.log('done')),
)
```

## Debug

Use `DEBUG=p-transform:*` environment variable.

## License

Apache-2.0

# API

## Classes

<dl>
<dt><a href="#OutOfOrder">OutOfOrder</a></dt>
<dd></dd>
</dl>

## Constants

<dl>
<dt><a href="#pipeline">pipeline</a></dt>
<dd><p>Promisified pipeline</p>
</dd>
</dl>

## Functions

<dl>
<dt><a href="#transform">transform(transform, end)</a></dt>
<dd><p>Shortcut to create a OutOfOrder with transform and end callback</p>
</dd>
<dt><a href="#passthrough">passthrough(spy, end)</a></dt>
<dd><p>Shortcut to create a passthrough OutOfOrder with spy and end callback</p>
</dd>
<dt><a href="#filter">filter(filter, end)</a></dt>
<dd><p>Shortcut to create a filter OutOfOrder with filter and end callback</p>
</dd>
</dl>

<a name="OutOfOrder"></a>

## OutOfOrder

**Kind**: global class

- [OutOfOrder](#OutOfOrder)
  - [new OutOfOrder(transform[, options])](#new_OutOfOrder_new)
  - [.duplex(endCallback)](#OutOfOrder+duplex) ⇒ [<code>Duplex</code>](https://nodejs.org/api/stream.html#class-streamduplex)

<a name="new_OutOfOrder_new"></a>

### new OutOfOrder(transform[, queueOptions])

OutOfOrder

| Param          | Type                  | Description                           |
| -------------- | --------------------- | ------------------------------------- |
| [transform]    | <code>function</code> | Transform.                            |
| [queueOptions] | <code>Object</code>   | Options forwarded to PQueue instance. |

<a name="OutOfOrder+duplex"></a>

### outOfOrder.duplex(end) ⇒ [<code>Duplex</code>](https://nodejs.org/api/stream.html#class-streamduplex)

Build Duplex.

**Kind**: instance method of [<code>OutOfOrder</code>](#OutOfOrder)
**Returns**: [<code>Duplex</code>](https://nodejs.org/api/stream.html#class-streamduplex)

| Param | Type                  |
| ----- | --------------------- |
| end   | <code>function</code> |

<a name="OutOfOrder+flushQueue"></a>

## pipeline

Promisified pipeline

**Kind**: global constant
<a name="transform"></a>

## transform(transform, end)

Shortcut to create a OutOfOrder with transform and end callback.

**Kind**: global function

| Param     | Type                  |
| --------- | --------------------- |
| transform | <code>function</code> |
| end       | <code>function</code> |

<a name="passthrough"></a>

## passthrough(spy, end)

Shortcut to create a passthrough OutOfOrder with spy and end callback.

**Kind**: global function

| Param | Type                  |
| ----- | --------------------- |
| spy   | <code>function</code> |
| end   | <code>function</code> |

<a name="filter"></a>

## filter(filter, end)

Shortcut to create a filter OutOfOrder with filter and end callback.

**Kind**: global function

| Param  | Type                  |
| ------ | --------------------- |
| filter | <code>function</code> |
| end    | <code>function</code> |
