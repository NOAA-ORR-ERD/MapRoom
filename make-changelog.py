#!/usr/bin/env python

import os,sys,re,os.path,time, subprocess
from cStringIO import StringIO
from datetime import date
from optparse import OptionParser
from string import Template
from distutils.version import StrictVersion

module=None

dateformat = "%m/%d/%Y"

versionre = "([0-9]+(\.[0-9]+)+([ab][0-9]+)?)"
versionre = "(([0-9]+)\.([0-9]+)(?:\.([0-9]+))?(([ab])([0-9]+))?)"

def findLatestChangeLogVersion(options):
    fh = open("ChangeLog")
    release_date = None
    versions = []
    codename = None
    for line in fh:
        if release_date is None:
            match = re.match('(\d+/\d+/\d+).*',line)
            if match:
                if options.verbose: print 'found date %s' % match.group(1)
                release_date = date.fromtimestamp(time.mktime(time.strptime(match.group(1),'%m/%d/%Y'))).strftime(dateformat)
        else:
            match = re.match('[Rr]eleased %s' % versionre,line)
            if match:
                if options.verbose: print 'found version %s' % match.group(1)
                version = match.group(1)
                versions.append(version)
    if release_date is None:
        release_date = date.today().strftime(dateformat)
    if not versions:
        version = "0.0"
    else:
        version = versions[0]
    return version, release_date, versions

def findChangeLogVersionForGit(options):
    fh = open("ChangeLog")
    release_date = date.today().strftime(dateformat)
    version = "0.0.0"
    codename = ""
    for line in fh:
        match = re.match('(\d+-\d+-\d+).*',line)
        if match:
            if options.verbose: print 'found date %s' % match.group(1)
            release_date = date.fromtimestamp(time.mktime(time.strptime(match.group(1),'%Y-%m-%d'))).strftime(dateformat)
        match = re.match('\s+\*\s*[Rr]eleased ([0-9]+\.[0-9]+(?:\.[0-9]+)?) \"(.+)\"',line)
        if match:
            if options.verbose: print 'found version %s' % match.group(1)
            version = match.group(1)
            codename = match.group(2)
            break
        release_date = None
    if release_date is None:
        release_date = date.today().strftime(dateformat)
    return version, release_date, codename

def findLatestInGit(options):
    version = StrictVersion("0.0")
    tags = subprocess.Popen(["git", "tag", "-l"], stdout=subprocess.PIPE).communicate()[0]
    for tag in tags.splitlines():
        match = re.match(r'%s$' % versionre, tag)
        if match:
            found = StrictVersion(match.group(1))
            if found > version:
                version = found
            if options.verbose: print "found %s, latest = %s" % (found, version)
    return str(version)

def next_version(tagged_version):
    t = tagged_version.split(".")
#    print t
    last = t[-1]
    try:
        last = str(int(last) + 1)
    except ValueError:
        for i in range(0, len(last) - 1):
            try:
                last = last[0:i+1] + str(int(last[i+1:]) + 1)
                break
            except ValueError:
                pass
    t[-1] = last
    return ".".join(t)

def getCurrentGitMD5s(tag, options):
    text = subprocess.Popen(["git", "rev-list", "%s..HEAD" % tag], stdout=subprocess.PIPE).communicate()[0]
    md5s = text.splitlines()
    return md5s

def getInitialsFromEmail(email):
    name, domain = email.split("@")
    names = name.split(".")
    initials = []
    for name in names:
        initials.append(name[0].upper())
    return "".join(initials)

def isImportantChangeLogLine(text):
    if text.startswith("Merge branch"):
        return False
    if text.startswith("updated ChangeLog & Version.py for"):
        return False
    return True

def getGitChangeLogSuggestions(tag, options):
    if options.verbose: print tag
    top = "HEAD"
    suggestions = []
    text = subprocess.Popen(["git", "log", "--pretty=format:%ae--%B", "%s..%s" % (tag, top)], stdout=subprocess.PIPE).communicate()[0]
    lines = text.splitlines()
    print lines
    first = True
    for line in lines:
        if first:
            if "--" in line and "@" in line:
                print line
                email, text = line.split("--", 1)
                initials = getInitialsFromEmail(email)
                if isImportantChangeLogLine(text):
                    suggestions.append("- %s %s" % (initials, text))
                first = False
        elif not line:
            first = True
    return suggestions

def getChangeLogBlock(git_version, module_version, options):
    new_block = []
    suggestions = getGitChangeLogSuggestions(git_version, options)
    now = date.today().strftime(dateformat)
    new_block.append(now)
    new_block.append("---------------------------")
    new_block.append("Released %s" % module_version)
    for line in suggestions:
        new_block.append(line)
    print "\n".join(new_block)
    return new_block

def prepend(filename, block):
    fh = open(filename)
    text = fh.read()
    fh.close()
    
    fh = open(filename, "w")
    fh.write("\n".join(block))
    fh.write("\n\n")
    fh.write(text)
    fh.close()

def replace(filename, block):
    fh = open(filename)
    current = []
    store = False
    for line in fh:
        if store:
            current.append(line)
        if not line.strip(): # skip until first blank line
            store = True
    fh.close()
    
    fh = open(filename, "w")
    fh.write("\n".join(block))
    fh.write("\n\n")
    fh.write("".join(current))
    fh.close()

if __name__=='__main__':
    usage="usage: %prog [-m module] [-o file] [-n variablename file] [-t template] [files...]"
    parser=OptionParser(usage=usage)
    parser.add_option("-v", "--verbose", action="store_true", help="print debugging info")
    parser.add_option("--dry-run", action="store_true", help="don't actually change anything")
    parser.add_option("--version", action="store_true", help="display current version and exit")
    parser.add_option("--next-version", action="store_true", help="display what would be the next version and exit")
    parser.add_option("-o", action="store", dest="outputfile", help="output filename", default="ChangeLog")
    (options, args) = parser.parse_args()

    tagged_version = findLatestInGit(options)
    if options.verbose:
        print "latest tagged in git: %s" % tagged_version
    if options.version:
        print tagged_version
        sys.exit()
    if options.next_version:
        print next_version(tagged_version)
        sys.exit()
    
    version, latest_date, versions = findLatestChangeLogVersion(options)
    print "latest from changelog: %s" % version
    print "all from changelog: %s" % str(versions)

    import importlib
    module = importlib.import_module("maproom.Version")
    print module
    print "module version: %s" % module.VERSION
    
    v_changelog = StrictVersion(version)
    print v_changelog
    v_module = StrictVersion(module.VERSION)
    print v_module
    v_tagged = StrictVersion(tagged_version)
    print v_tagged
    
    if v_module > v_changelog:
        print "adding to ChangeLog!"
        block = getChangeLogBlock(tagged_version, module.VERSION, options)
        if not options.dry_run:
            prepend(options.outputfile, block)
    elif v_module == v_changelog and v_module > v_tagged:
        print "replacing ChangeLog entry!"
        block = getChangeLogBlock(tagged_version, module.VERSION, options)
        if not options.dry_run:
            replace(options.outputfile, block)
    else:
            print "unhandled version differences..."
