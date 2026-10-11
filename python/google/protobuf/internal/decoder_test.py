#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Tests for the pure Python decoder's recursion limit.

The pure Python decoder walks nested sub-messages, groups, map entries,
MessageSet items and unknown groups with Python recursion, so a hostile payload
of nested START_GROUP tags (or of nested length-delimited sub-messages) used to
exhaust the interpreter stack with a RecursionError.  The decoder now carries a
depth counter and raises message.DecodeError once DEFAULT_RECURSION_LIMIT
levels are exceeded.
"""

try:
  import unittest2 as unittest  #PY26
except ImportError:
  import unittest

import six

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import map_unittest_pb2
from google.protobuf import message
from google.protobuf import message_factory
from google.protobuf import text_format
from google.protobuf import unittest_mset_pb2
from google.protobuf import unittest_mset_wire_format_pb2
from google.protobuf import unittest_pb2
from google.protobuf.internal import api_implementation
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import testing_refleaks
from google.protobuf.internal import wire_format


# The depth guard lives in the pure Python decoder only; the C++ backend has its
# own (differently worded) limit, so the message level tests only run there.
_IS_PURE_PYTHON = api_implementation.Type() == 'python'

# A stream of identical START_GROUP tags: 0o23 == 0x13 packs field number 2 with
# wire type 3 (WIRETYPE_START_GROUP), so every byte opens one more group.
_NESTED_START_GROUPS = b'\023'


def _LengthDelimited(field_number, payload):
  """Encodes one length-delimited field."""
  return (encoder.TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
          + encoder._VarintBytes(len(payload)) + payload)


def _MessageSetItem(type_id, payload, prefix=b''):
  """Encodes one MessageSet item group carrying `payload` for `type_id`."""
  return (encoder.TagBytes(1, wire_format.WIRETYPE_START_GROUP) + prefix +
          encoder.TagBytes(2, wire_format.WIRETYPE_VARINT) +
          encoder._VarintBytes(type_id) +
          _LengthDelimited(3, payload) +
          encoder.TagBytes(1, wire_format.WIRETYPE_END_GROUP))


@testing_refleaks.TestCase
class DecoderRecursionTest(unittest.TestCase):

  def tearDown(self):
    # Make sure a test that raised the limit cannot leak it into another test.
    decoder.SetRecursionLimit(decoder.DEFAULT_RECURSION_LIMIT)

  def assertTooManyLevels(self, context):
    self.assertIn('Error parsing message', str(context.exception))
    self.assertIn('too many levels of nesting', str(context.exception))

  def GenerateNestedProto(self, n):
    """Serializes a TestRecursiveMessage nested n levels deep."""
    msg = unittest_pb2.TestRecursiveMessage()
    sub = msg
    for _ in range(n):
      sub = sub.a
    sub.i = 1
    return msg.SerializeToString()

  def GenerateRepeatedNestedProto(self, n):
    """Serializes a NestedTestAllTypes nested n levels deep via repeated_child."""
    msg = unittest_pb2.NestedTestAllTypes()
    sub = msg
    for _ in range(n):
      sub = sub.repeated_child.add()
    sub.payload.optional_int32 = 1
    return msg.SerializeToString()

  def GenerateNestedGroupProto(self, cycles):
    """Serializes a TestMutualRecursionA whose recursion runs through a group.

    Each cycle costs four levels of decoder recursion: the SubGroup group, its
    sub_message, that message's `b`, and `b`'s `a`.  The innermost `bb` adds one
    more, so the deepest level reached is 4 * cycles + 1.
    """
    msg = unittest_pb2.TestMutualRecursionA()
    sub = msg
    for _ in range(cycles):
      sub = sub.subgroup.sub_message.b.a
    sub.bb.optional_int32 = 1
    return msg.SerializeToString()

  def GenerateNestedMessageSet(self, n):
    """Builds TestMessageSet wire bytes nested n MessageSet items deep.

    TestMessageSetExtension1 carries a `recursive` field back to TestMessageSet,
    so every level re-enters MessageSetItemDecoder's DecodeItem.  Each level
    costs two levels of decoder recursion (the item's extension message and its
    `recursive` sub-message).  The bytes are built directly rather than by
    serializing a live message, because serializing a deeply nested message
    would itself recurse.
    """
    type_id = (
        unittest_mset_pb2.TestMessageSetExtension1.message_set_extension.number)
    # The innermost extension only sets i = 1.
    ext = encoder.TagBytes(15, wire_format.WIRETYPE_VARINT) + (
        encoder._VarintBytes(1))
    data = _MessageSetItem(type_id, ext)
    for _ in range(n - 1):
      data = _MessageSetItem(type_id, _LengthDelimited(16, data))
    return data

  def GenerateNestedMap(self, n):
    """Builds TestRecursiveMapMessage wire bytes nested n map entries deep.

    TestRecursiveMapMessage.a is a map<string, TestRecursiveMapMessage>, so each
    level descends through MapDecoder's DecodeMap.  Each level costs two levels
    of decoder recursion (the map entry and its value message).
    """
    data = b''
    for _ in range(n):
      entry = _LengthDelimited(1, b'x') + _LengthDelimited(2, data)
      data = _LengthDelimited(1, entry)
    return data

  def testDefaultRecursionLimit(self):
    self.assertEqual(100, decoder.DEFAULT_RECURSION_LIMIT)
    self.assertEqual(decoder.DEFAULT_RECURSION_LIMIT, decoder._recursion_limit)

  def test_decode_unknown_group_field_too_many_levels(self):
    # 5 MB of nested START_GROUP tags: without the depth guard this recurses
    # until the interpreter stack is exhausted.
    data = memoryview(_NESTED_START_GROUPS * 5000000)
    six.assertRaisesRegex(
        self,
        message.DecodeError,
        'Error parsing message',
        decoder._DecodeUnknownField,
        data,
        1,
        wire_format.WIRETYPE_START_GROUP,
    )

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testUnknownGroupTooManyLevelsWhileParsing(self):
    # The same payload reached through the public parse entry point: TestAllTypes
    # has no group with field number 2, so every tag lands on the unknown field
    # path.
    msg = unittest_pb2.TestAllTypes()
    with self.assertRaises(message.DecodeError) as context:
      msg.ParseFromString(_NESTED_START_GROUPS * 5000)
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testUnknownGroupWithinLimit(self):
    start = encoder.TagBytes(2, wire_format.WIRETYPE_START_GROUP)
    end = encoder.TagBytes(2, wire_format.WIRETYPE_END_GROUP)
    data = start * 50 + end * 50
    msg = unittest_pb2.TestAllTypes()
    msg.ParseFromString(data)
    self.assertEqual(data, msg.SerializeToString())

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testSucceedOkSizedProto(self):
    msg = unittest_pb2.TestRecursiveMessage()
    msg.ParseFromString(self.GenerateNestedProto(100))
    # Walk the whole chain back down to prove 100 levels really were parsed.
    sub = msg
    for _ in range(100):
      sub = sub.a
    self.assertEqual(1, sub.i)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testAssertOversizeProto(self):
    msg = unittest_pb2.TestRecursiveMessage()
    with self.assertRaises(message.DecodeError) as context:
      msg.ParseFromString(self.GenerateNestedProto(101))
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testHostileNestedMessagePayload(self):
    # A hostile payload nested 1000 levels through the singular `a` field,
    # deep enough to exhaust the interpreter stack without the guard.
    data = encoder.TagBytes(2, wire_format.WIRETYPE_VARINT) + (
        encoder._VarintBytes(1))
    for _ in range(1000):
      data = _LengthDelimited(1, data)
    with self.assertRaises(message.DecodeError) as context:
      unittest_pb2.TestRecursiveMessage().ParseFromString(data)
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testSucceedOversizeProtoWithRaisedLimit(self):
    data = self.GenerateNestedProto(101)
    decoder.SetRecursionLimit(310)
    try:
      msg = unittest_pb2.TestRecursiveMessage()
      msg.ParseFromString(data)
    finally:
      decoder.SetRecursionLimit(decoder.DEFAULT_RECURSION_LIMIT)
    self.assertTrue(msg.HasField('a'))
    # And the default limit is back in force afterwards.
    with self.assertRaises(message.DecodeError):
      unittest_pb2.TestRecursiveMessage().ParseFromString(data)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testRepeatedSubMessageTooManyLevels(self):
    # MessageDecoder's repeated branch is a separate recursion site from the
    # singular one exercised above.
    msg = unittest_pb2.NestedTestAllTypes()
    msg.ParseFromString(self.GenerateRepeatedNestedProto(20))
    sub = msg
    for _ in range(20):
      self.assertEqual(1, len(sub.repeated_child))
      sub = sub.repeated_child[0]
    self.assertEqual(1, sub.payload.optional_int32)

    with self.assertRaises(message.DecodeError) as context:
      unittest_pb2.NestedTestAllTypes().ParseFromString(
          self.GenerateRepeatedNestedProto(110))
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testHostileNestedRepeatedMessagePayload(self):
    # A hostile payload nested 1000 levels through repeated_child.
    data = b''
    for _ in range(1000):
      data = _LengthDelimited(3, data)
    with self.assertRaises(message.DecodeError) as context:
      unittest_pb2.NestedTestAllTypes().ParseFromString(data)
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testGroupFieldTooManyLevels(self):
    msg = unittest_pb2.TestMutualRecursionA()
    # 20 cycles == 81 levels, comfortably under the limit.
    msg.ParseFromString(self.GenerateNestedGroupProto(20))
    self.assertTrue(msg.HasField('subgroup'))
    sub = msg
    for _ in range(20):
      sub = sub.subgroup.sub_message.b.a
    self.assertEqual(1, sub.bb.optional_int32)

    # 30 cycles == 121 levels.
    with self.assertRaises(message.DecodeError) as context:
      unittest_pb2.TestMutualRecursionA().ParseFromString(
          self.GenerateNestedGroupProto(30))
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testHostileNestedGroupPayload(self):
    # A hostile payload of 300 cycles (1200 levels) through the SubGroup group:
    # SubGroup { sub_message (3) { b (1) { a (1) { ... } } } }.
    start = encoder.TagBytes(2, wire_format.WIRETYPE_START_GROUP)
    end = encoder.TagBytes(2, wire_format.WIRETYPE_END_GROUP)
    data = b''
    for _ in range(300):
      data = start + _LengthDelimited(
          3, _LengthDelimited(1, _LengthDelimited(1, data))) + end
    with self.assertRaises(message.DecodeError) as context:
      unittest_pb2.TestMutualRecursionA().ParseFromString(data)
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testMessageSetItemTooManyLevels(self):
    # MessageSetItemDecoder.DecodeItem is registered straight into
    # _decoders_by_tag, so InternalParse calls it with current_depth like any
    # other field decoder -- it must accept the argument and count the
    # extension message it parses, or the MessageSet wire format is both a
    # TypeError and an unguarded recursion path.
    ext_handle = unittest_mset_pb2.TestMessageSetExtension1.message_set_extension
    msg = unittest_mset_wire_format_pb2.TestMessageSet()
    msg.ParseFromString(self.GenerateNestedMessageSet(20))  # 39 levels.
    sub = msg
    for _ in range(19):
      self.assertTrue(sub.HasExtension(ext_handle))
      sub = sub.Extensions[ext_handle].recursive
    self.assertEqual(1, sub.Extensions[ext_handle].i)

    # 500 items == 999 levels: exhausts the interpreter stack without the guard.
    with self.assertRaises(message.DecodeError) as context:
      unittest_mset_wire_format_pb2.TestMessageSet().ParseFromString(
          self.GenerateNestedMessageSet(500))
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testMessageSetItemUnknownGroupTooManyLevels(self):
    # Fields inside a MessageSet item other than type_id/message are skipped
    # with SkipField, whose group skipping recurses on its own.
    type_id = (
        unittest_mset_pb2.TestMessageSetExtension1.message_set_extension.number)
    ext_handle = unittest_mset_pb2.TestMessageSetExtension1.message_set_extension
    payload = unittest_mset_pb2.TestMessageSetExtension1(i=7).SerializeToString()
    start = encoder.TagBytes(4, wire_format.WIRETYPE_START_GROUP)
    end = encoder.TagBytes(4, wire_format.WIRETYPE_END_GROUP)
    unknown = start * 3 + b'\x08\x01' + end * 3
    msg = unittest_mset_wire_format_pb2.TestMessageSet()
    msg.ParseFromString(_MessageSetItem(type_id, payload, prefix=unknown))
    self.assertEqual(7, msg.Extensions[ext_handle].i)

    # A MessageSet item carrying an unknown field that opens an unbounded
    # series of groups.
    data = encoder.TagBytes(1, wire_format.WIRETYPE_START_GROUP) + start * 100000
    with self.assertRaises(message.DecodeError) as context:
      unittest_mset_wire_format_pb2.TestMessageSet().ParseFromString(data)
    self.assertTooManyLevels(context)

  @unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
  def testMapEntryTooManyLevels(self):
    msg = map_unittest_pb2.TestRecursiveMapMessage()
    msg.ParseFromString(self.GenerateNestedMap(10))  # 20 levels.
    sub = msg
    for _ in range(10):
      self.assertIn('x', sub.a)
      sub = sub.a['x']
    self.assertEqual(0, len(sub.a))

    # 1000 entries == 2000 levels: exhausts the interpreter stack without the
    # guard.
    with self.assertRaises(message.DecodeError) as context:
      map_unittest_pb2.TestRecursiveMapMessage().ParseFromString(
          self.GenerateNestedMap(1000))
    self.assertTooManyLevels(context)


@unittest.skipIf(not _IS_PURE_PYTHON, 'the depth guard is pure Python only')
@testing_refleaks.TestCase
class RepeatedGroupDecoderRecursionTest(unittest.TestCase):
  """Covers GroupDecoder's repeated branch, its own recursion site.

  The test protos have no self-recursive *repeated* group, so the message type
  is built from a descriptor at runtime the way OversizeProtosTest in
  message_test.py does, rather than by adding a .proto.
  """

  @classmethod
  def setUpClass(cls):
    # Reference cycles between DescriptorPool and Message classes are not
    # detected, so build the class exactly once to keep the refleak checker
    # happy (same reasoning as message_test.OversizeProtosTest).
    file_desc = """
      name: "g/recursive_group.proto"
      package: "g"
      message_type {
        name: "RecursiveGroup"
        field {
          name: "child"
          number: 1
          label: LABEL_REPEATED
          type: TYPE_GROUP
          type_name: "RecursiveGroup"
        }
        field {
          name: "i"
          number: 2
          label: LABEL_OPTIONAL
          type: TYPE_INT32
        }
      }
    """
    pool = descriptor_pool.DescriptorPool()
    desc = descriptor_pb2.FileDescriptorProto()
    text_format.Parse(file_desc, desc)
    pool.Add(desc)
    cls.proto_cls = message_factory.MessageFactory(pool).GetPrototype(
        pool.FindMessageTypeByName('g.RecursiveGroup'))

  def tearDown(self):
    decoder.SetRecursionLimit(decoder.DEFAULT_RECURSION_LIMIT)

  def GenerateNestedRepeatedGroup(self, n):
    msg = self.proto_cls()
    sub = msg
    for _ in range(n):
      sub = sub.child.add()
    sub.i = 1
    return msg.SerializeToString()

  def testRepeatedGroupWithinLimit(self):
    msg = self.proto_cls()
    msg.ParseFromString(self.GenerateNestedRepeatedGroup(100))
    sub = msg
    for _ in range(100):
      self.assertEqual(1, len(sub.child))
      sub = sub.child[0]
    self.assertEqual(1, sub.i)

  def testRepeatedGroupTooManyLevels(self):
    data = self.GenerateNestedRepeatedGroup(101)
    with self.assertRaises(message.DecodeError) as context:
      self.proto_cls().ParseFromString(data)
    self.assertIn('Error parsing message', str(context.exception))
    self.assertIn('too many levels of nesting', str(context.exception))

    # Raising the limit lets the same payload through.
    decoder.SetRecursionLimit(310)
    try:
      self.proto_cls().ParseFromString(data)
    finally:
      decoder.SetRecursionLimit(decoder.DEFAULT_RECURSION_LIMIT)

  def testHostileNestedRepeatedGroupPayload(self):
    # A hostile payload of 1000 nested `child` groups, deep enough to exhaust
    # the interpreter stack without the guard.
    data = (encoder.TagBytes(1, wire_format.WIRETYPE_START_GROUP) * 1000 +
            encoder.TagBytes(1, wire_format.WIRETYPE_END_GROUP) * 1000)
    with self.assertRaises(message.DecodeError) as context:
      self.proto_cls().ParseFromString(data)
    self.assertIn('too many levels of nesting', str(context.exception))


if __name__ == '__main__':
  unittest.main()
